# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
from typing import Dict
import asyncio

import ray
from src.actors.opus_to_pcm_actor import OpusToPcmConverterRayActor
from src.actors.vpr_actor import VprRayActor
from src.actors.itn_actor import ItnRayActor
from src.actors.silero_vad_actor import VadRayActor
from src.client_session import ClientSession
from src.configs.config import config
from src.definitions import InitialSpeakerRequest
# 导入错误处理相关模块
from src.exception.error_codes import (
    ErrorCode,
    ServiceError,
)
from src.exception.exception_handlers import (
    handle_exceptions,
)
from src.logger.logger_adapter import logger


class SessionManager:
    """管理所有会话的生命周期和Ray Actor资源的分配（安全加固版）"""

    def __init__(self):
        self.logger = logger
        self.logger.info("Initializing SessionManager and Ray Actor Pools...")

        # 初始化资源池
        self._free_itn_actors = []
        self._free_opus_actors = []
        self._free_vad_actors = []
        self._free_vpr_actors = []
        self._active_sessions: Dict[str, ClientSession] = {}

        # 全局 HTTP 并发控制（关键优化）
        # 限制同时进行的 HTTP 请求数量，防止过载
        self._http_semaphore = asyncio.Semaphore(50)  # 最多50个并发HTTP请求
        self.logger.info("Global HTTP semaphore initialized with limit=50")

        # 启动 HTTP 监控
        self._start_http_monitoring()

        try:
            self._init_actor_pools()
            # 验证Actor数量
            self._verify_actor_counts()
        except Exception as e:
            self.logger.critical(f"SessionManager initialization failed: {e}", exc_info=True)
            # 清理已创建的资源
            self._cleanup_partial_actors()
            raise ServiceError.from_exception(ErrorCode.SYSTEM_ERROR, e)

    def _start_http_monitoring(self):
        """启动 HTTP 性能监控"""
        try:
            import asyncio
            from src.utils.http_metrics import get_metrics_collector

            async def _init_monitoring():
                collector = await get_metrics_collector()
                await collector.start_monitoring(interval=60)  # 每60秒输出一次统计

            # 在事件循环中启动监控
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(_init_monitoring())
                else:
                    loop.run_until_complete(_init_monitoring())
            except RuntimeError:
                # 如果还没有事件循环，稍后再启动
                self.logger.warning("Event loop not ready, HTTP monitoring will start later")
        except Exception as e:
            self.logger.warning(f"Failed to start HTTP monitoring: {e}")

    def get_http_semaphore(self) -> asyncio.Semaphore:
        """获取全局 HTTP 并发控制信号量"""
        return self._http_semaphore

    @handle_exceptions
    async def create_session(self, websocket, initial_request: InitialSpeakerRequest) -> ClientSession:
        vpr_actor = self._free_vpr_actors.pop()
        vad_actor = self._free_vad_actors.pop()
        itn_actor = self._free_itn_actors.pop()
        opus_actor = None
        if initial_request.audioFormat.compress == "opus":
            opus_actor = self._free_opus_actors.pop()

        # 4. 创建会话并加入活跃会话列表
        session = ClientSession(
            initial_request=initial_request,
            websocket=websocket,
            opus_actor=opus_actor,
            vad_actor=vad_actor,
            vpr_actor=vpr_actor,
            itn_actor=itn_actor,
            session_manager=self,
        )

        # 验证会话ID唯一性
        if session.sessionId in self._active_sessions:
            raise ServiceError(
                error_code=ErrorCode.SESSION_CONFLICT,
                error_msg=f"Session ID {session.sessionId} already exists",
                details="Duplicate session ID detected",
            )

        self._active_sessions[session.sessionId] = session
        await session.start()
        self.logger.info(f"Session {session.sessionId} created and assigned resources successfully")
        return session

    @handle_exceptions
    async def remove_session(self, session: ClientSession):
        """移除会话（加固：健壮的Actor归还、空值检查、锁保护）"""
        # 1. 基础验证
        if not session:
            self.logger.warning("Attempted to remove null session")
            return

        sessionId = session.sessionId

        # 2. 加锁操作会话和资源池
        if sessionId not in self._active_sessions:
            raise ServiceError(
                error_code=ErrorCode.SESSION_NOT_FOUND,
                error_msg=f"Session {sessionId} not found in active sessions",
                retryable=False,
            )

        self.logger.info(f"Removing session {sessionId}.")

        # 3. 停止会话
        try:
            await session.stop()
        except Exception as e:
            self.logger.error(f"Failed to stop session {sessionId}: {e}", exc_info=True)
            # 即使停止失败，也要继续清理，避免资源泄漏

        # 4. 从活跃会话中移除
        del self._active_sessions[sessionId]

        opus_actor = session.get_opus_actor()
        vad_actor = session.get_vad_actor()
        vpr_actor = session.get_vpr_actor()
        itn_actor = session.get_itn_actor()

        # 归还Opus Actor（无需重置，先检查非空）
        if opus_actor:
            self._free_opus_actors.append(opus_actor)
        self.logger.info(f"removed opus actor successfully Opus={len(self._free_opus_actors)}")

        if vad_actor:
            self._free_vad_actors.append(vad_actor)
        self.logger.info(f"removed vad actor successfully VAD={len(self._free_vad_actors)}")

        if vpr_actor:
            self._free_vpr_actors.append(vpr_actor)
        self.logger.info(f"removed vpr actor successfully VPR={len(self._free_vpr_actors)}")

        if itn_actor:
            self._free_itn_actors.append(itn_actor)
        self.logger.info(f"removed vpr actor successfully ITN={len(self._free_itn_actors)}")


        
    def get_all_active_sessions(self):
        """获取所有活跃会话"""
        return self._active_sessions.values()

    def get_all_active_count(self):
        """获取所有活跃会话"""
        return len(self._active_sessions.values())

    @handle_exceptions
    async def stop(self):
        """停止管理器（加固：完整的资源清理）"""
        self.logger.info("Stopping SessionManager and cleaning up resources...")

        # 1. 停止所有活跃会话
        session_ids = list(self._active_sessions.keys())
        for sessionId in session_ids:
            try:
                await self.remove_session(self._active_sessions[sessionId])
            except Exception as e:
                self.logger.error(f"Failed to remove session {sessionId} during shutdown: {e}", exc_info=True)

        # 2. 终止所有Actor
        self._cleanup_partial_actors()

        # 3. 清空资源池和会话
        self._free_opus_actors.clear()
        self._free_vad_actors.clear()
        self._free_vpr_actors.clear()
        self._free_itn_actors.clear()
        self._active_sessions.clear()

        self.logger.info("SessionManager stopped and all resources cleaned up")

    def _init_actor_pools(self):

        self.logger.info(f"Waiting for all itn actors to initialize... {config.itn_path}")
        itn_actors = [ItnRayActor(config) for _ in range(int(config.max_connections))]
        self._free_itn_actors = itn_actors[:]
        self.logger.info("All itn actors are ready.")

        self.logger.info(f"Waiting for all vpr actors to initialize... {config.vpr_path}")
        vpr_actors = [VprRayActor(config) for _ in range(int(config.max_connections))]
        self._free_vpr_actors = vpr_actors[:]
        self.logger.info("All vpr actors are ready.")

        # Opus解码Actor
        self.logger.info("Waiting for all opus actors to initialize...")
        opus_convertor_actors = [OpusToPcmConverterRayActor() for _ in range(config.max_connections)]
        self._free_opus_actors = opus_convertor_actors[:]
        self.logger.info("All opus convertor actors are ready.")

        self.logger.info(f"Waiting for all vad actors to initialize... {config.vad_path}")
        vad_actors = [VadRayActor(config.vad_path) for _ in range(config.max_connections)]
        self._free_vad_actors = vad_actors[:]
        self.logger.info("All vad actors are ready.")

        

    def _verify_actor_counts(self):
        """验证Actor数量（加固：异常时抛出明确错误）"""
        try:
            actor_classes = [a["ActorClassName"] for a in ray.state.actors().values()]
            expected = config.max_connections

            counts = {
                #"OPUS": actor_classes.count("OpusToPcmConverterRayActor"),
                #"VAD": actor_classes.count("VadRayActor"),
            }

            #self.logger.info(f"init actors OPUS: {counts['OPUS']}")

            mismatches = [f"{k}={v}" for k, v in counts.items() if v != expected]
            if mismatches:
                raise ServiceError(
                    error_code=ErrorCode.SERVICE_UNAVAILABLE,
                    error_msg="Actor count mismatch with configuration",
                    details=f"Expected: {expected}, Actual: {', '.join(mismatches)}",
                )

        except Exception as e:
            raise ServiceError.from_exception(ErrorCode.SYSTEM_ERROR, e)

    def _cleanup_partial_actors(self):
        """清理部分初始化的Actor（加固：防止资源泄漏）"""
        self.logger.warning("Cleaning up partially initialized actors...")
        all_actors = [
            self._free_opus_actors,
            self._free_vad_actors,
            self._free_vpr_actors,
            self._free_itn_actors,
        ]
        # 清理所有 Actor
        for actor_class in all_actors:
            for actor in actor_class:
                try:
                    ray.kill(actor)
                except Exception as e:
                    self.logger.debug(f"尝试终止 Ray actor 失败: {str(e)}")

    def _validate_session_id(self, sessionId: str) -> bool:
        """验证会话ID合法性（加固：防止无效会话操作）"""
        if (
                not sessionId
                or not isinstance(sessionId, str)
                or len(sessionId.strip()) == 0
        ):
            self.logger.error(f"Invalid session ID: {sessionId}")
            return False
        return True




