"""异步 HTTP 客户端模块 - 使用 httpx 实现高性能并发请求"""
import asyncio
import base64
import hashlib
import hmac
import json
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

try:
    import httpx
except ImportError:
    raise ImportError("httpx is required. Install it with: pip install httpx")

from src.configs.config import config
from src.utils.crypt_util import decrypt_secret
from src.logger.logger_adapter import logger

# 延迟导入避免循环依赖
_metrics_collector = None


async def _get_metrics_collector():
    """延迟获取 metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        try:
            from src.utils.http_metrics import get_metrics_collector
            _metrics_collector = await get_metrics_collector()
        except Exception as e:
            logger.warning(f"Failed to initialize metrics collector: {e}")
    return _metrics_collector


class AsyncHttpClient:
    """全局异步 HTTP 客户端，支持连接池复用和并发控制"""

    _instance: Optional['AsyncHttpClient'] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._is_initialized = False

    @classmethod
    async def get_instance(cls) -> 'AsyncHttpClient':
        """获取全局单例实例"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.initialize()
        return cls._instance

    async def initialize(self):
        """初始化 HTTP 客户端和连接池"""
        if self._is_initialized:
            return

        # 配置连接池参数
        limits = httpx.Limits(
            max_connections=100,  # 最大连接数
            max_keepalive_connections=50,  # 保持活跃的连接数
            keepalive_expiry=30.0,  # 连接保持时间（秒）
        )

        # 配置超时
        timeout = httpx.Timeout(
            connect=2.0,  # 连接超时
            read=5.0,     # 读取超时
            write=5.0,    # 写入超时
            pool=2.0,     # 连接池超时
        )

        self._client = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            http2=True,  # 启用 HTTP/2
            follow_redirects=False,
        )

        self._is_initialized = True
        logger.info(
            f"AsyncHttpClient initialized with "
            f"max_connections={limits.max_connections}, "
            f"keepalive={limits.max_keepalive_connections}"
        )

    async def close(self):
        """关闭客户端和连接池"""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._is_initialized = False
            logger.info("AsyncHttpClient closed")

    async def post(
        self,
        url: str,
        headers: Dict[str, str],
        data: Dict[str, Any],
        timeout: Optional[float] = None,
        retry_times: int = 3,
        retry_delay: float = 0.1,
        semaphore: Optional[asyncio.Semaphore] = None,
        api_type: str = "unknown",
    ) -> Optional[Dict[str, Any]]:
        """
        异步 POST 请求，支持重试和并发控制

        Args:
            url: 请求URL
            headers: 请求头
            data: 请求数据
            timeout: 超时时间（秒），None表示使用默认
            retry_times: 重试次数
            retry_delay: 重试间隔（秒）
            semaphore: 并发控制信号量（可选）
            api_type: API类型标识（qwen3_asr, speaker_omni等）

        Returns:
            响应数据字典，失败返回 None
        """
        if not self._is_initialized:
            await self.initialize()

        # 使用信号量控制并发
        if semaphore:
            async with semaphore:
                return await self._do_post(url, headers, data, timeout, retry_times, retry_delay, api_type)
        else:
            return await self._do_post(url, headers, data, timeout, retry_times, retry_delay, api_type)

    async def _do_post(
        self,
        url: str,
        headers: Dict[str, str],
        data: Dict[str, Any],
        timeout: Optional[float],
        retry_times: int,
        retry_delay: float,
        api_type: str = "unknown",
    ) -> Optional[Dict[str, Any]]:
        """实际执行 POST 请求的内部方法"""
        last_error = None
        start_time = time.time()
        success = False
        is_timeout = False

        for attempt in range(retry_times):
            try:
                request_timeout = timeout if timeout else 5.0
                attempt_start = time.time()

                # 注意：使用 content 而不是 json，因为需要手动序列化来匹配原始实现
                response = await self._client.post(
                    url,
                    headers=headers,
                    content=json.dumps(data),
                    timeout=request_timeout,
                )

                request_latency = time.time() - attempt_start

                # 检查 HTTP 状态码
                if response.status_code != 200:
                    logger.warning(
                        f"[{api_type}] HTTP {response.status_code}, "
                        f"latency={request_latency:.3f}s, "
                        f"attempt {attempt + 1}/{retry_times}"
                    )
                    if attempt < retry_times - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    return None

                result = response.json()

                # 检查业务状态码
                if result.get('result', {}).get('code') == '0':
                    success = True
                    total_latency = time.time() - start_time

                    # 记录成功的请求耗时（使用 DEBUG 级别，避免刷屏）
                    logger.debug(
                        f"[{api_type}] Success, "
                        f"latency={request_latency:.3f}s, "
                        f"total={total_latency:.3f}s, "
                        f"attempt={attempt + 1}"
                    )

                    collector = await _get_metrics_collector()
                    if collector:
                        collector.record_request(url, total_latency, success=True)
                    return result['result']['content'][0]
                else:
                    logger.warning(
                        f"[{api_type}] Business error: {result.get('result', {})}, "
                        f"latency={request_latency:.3f}s"
                    )
                    return None

            except httpx.TimeoutException as e:
                last_error = e
                is_timeout = True
                request_latency = time.time() - attempt_start
                logger.warning(
                    f"[{api_type}] Timeout, "
                    f"latency={request_latency:.3f}s, "
                    f"attempt {attempt + 1}/{retry_times}"
                )
                if attempt < retry_times - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

            except httpx.ConnectError as e:
                last_error = e
                request_latency = time.time() - attempt_start
                logger.warning(
                    f"[{api_type}] ConnectError, "
                    f"latency={request_latency:.3f}s, "
                    f"attempt {attempt + 1}/{retry_times}: {e}"
                )
                if attempt < retry_times - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(f"HTTP status error from {url}: {e}")
                return None

            except json.JSONDecodeError as e:
                last_error = e
                logger.error(f"Invalid JSON response from {url}: {e}")
                return None

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error calling {url}: {e}", exc_info=True)
                return None

        # 所有重试都失败，记录失败的请求
        total_latency = time.time() - start_time
        collector = await _get_metrics_collector()
        if collector:
            collector.record_request(url, total_latency, success=False, is_timeout=is_timeout)

        logger.error(
            f"[{api_type}] Failed, "
            f"total_latency={total_latency:.3f}s, "
            f"attempts={retry_times}, "
            f"last_error={last_error}"
        )
        return None


# 全局客户端实例（延迟初始化）
_global_client: Optional[AsyncHttpClient] = None


async def get_http_client() -> AsyncHttpClient:
    """获取全局 HTTP 客户端实例"""
    global _global_client
    if _global_client is None:
        _global_client = await AsyncHttpClient.get_instance()
    return _global_client


async def close_http_client():
    """关闭全局 HTTP 客户端"""
    global _global_client
    if _global_client:
        await _global_client.close()
        _global_client = None
