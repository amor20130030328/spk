"""HTTP 请求性能监控模块"""
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

from src.logger.logger_adapter import logger


@dataclass
class HttpMetrics:
    """HTTP 请求性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    total_latency: float = 0.0  # 总延迟（秒）
    min_latency: float = float('inf')
    max_latency: float = 0.0

    # 按URL统计
    url_stats: Dict[str, Dict] = field(default_factory=lambda: defaultdict(lambda: {
        'count': 0,
        'success': 0,
        'failed': 0,
        'total_latency': 0.0
    }))

    def record_request(
        self,
        url: str,
        latency: float,
        success: bool,
        is_timeout: bool = False
    ):
        """记录一次请求"""
        self.total_requests += 1
        self.total_latency += latency

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        if is_timeout:
            self.timeout_requests += 1

        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)

        # URL 级别统计
        stats = self.url_stats[url]
        stats['count'] += 1
        stats['total_latency'] += latency
        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1

    def get_avg_latency(self) -> float:
        """获取平均延迟"""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests

    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests * 100

    def get_summary(self) -> str:
        """获取统计摘要"""
        return (
            f"HTTP Metrics: "
            f"total={self.total_requests}, "
            f"success={self.successful_requests}, "
            f"failed={self.failed_requests}, "
            f"timeout={self.timeout_requests}, "
            f"success_rate={self.get_success_rate():.1f}%, "
            f"avg_latency={self.get_avg_latency():.3f}s, "
            f"min={self.min_latency:.3f}s, "
            f"max={self.max_latency:.3f}s"
        )

    def get_url_summary(self) -> str:
        """获取按 URL 的统计摘要"""
        if not self.url_stats:
            return "No URL stats available"

        lines = ["URL Statistics:"]
        for url, stats in self.url_stats.items():
            avg_latency = stats['total_latency'] / stats['count'] if stats['count'] > 0 else 0
            success_rate = stats['success'] / stats['count'] * 100 if stats['count'] > 0 else 0
            lines.append(
                f"  {url}: count={stats['count']}, "
                f"success_rate={success_rate:.1f}%, "
                f"avg_latency={avg_latency:.3f}s"
            )
        return "\n".join(lines)


class HttpMetricsCollector:
    """全局 HTTP 指标收集器"""

    _instance: Optional['HttpMetricsCollector'] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.metrics = HttpMetrics()
        self._monitor_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

    @classmethod
    async def get_instance(cls) -> 'HttpMetricsCollector':
        """获取全局单例"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_request(
        self,
        url: str,
        latency: float,
        success: bool,
        is_timeout: bool = False
    ):
        """记录请求"""
        self.metrics.record_request(url, latency, success, is_timeout)

    async def start_monitoring(self, interval: int = 60):
        """启动定期监控日志输出"""
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("HTTP metrics monitoring started")

    async def stop_monitoring(self):
        """停止监控"""
        self._is_monitoring = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("HTTP metrics monitoring stopped")

    async def _monitor_loop(self, interval: int):
        """监控循环"""
        try:
            while self._is_monitoring:
                await asyncio.sleep(interval)
                self.log_summary()
        except asyncio.CancelledError:
            pass

    def log_summary(self):
        """输出统计摘要到日志"""
        logger.info(self.metrics.get_summary())
        logger.info(self.metrics.get_url_summary())

    def reset(self):
        """重置统计数据"""
        self.metrics = HttpMetrics()
        logger.info("HTTP metrics reset")


# 全局收集器实例
_global_collector: Optional[HttpMetricsCollector] = None


async def get_metrics_collector() -> HttpMetricsCollector:
    """获取全局指标收集器"""
    global _global_collector
    if _global_collector is None:
        _global_collector = await HttpMetricsCollector.get_instance()
    return _global_collector
