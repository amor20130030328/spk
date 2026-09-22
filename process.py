from __future__ import annotations
import asyncio
import json
import os
import logging
import threading
import sys
import ray
import functools

from common.util import logutil
from common.util.ws_util import WorkMessage, WsConnector
from fastapi import WebSocket, WebSocketDisconnect
from src.context.context_helper import ContextHelper
from src.definitions import InitialSpeakerRequest
from src.logger.logger_adapter import logger
from src.session_manager import SessionManager



current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class AsyncIterableWebSocketAdapter:
  
    def __init__(self, websocket: WebSocket):
        self._ws = websocket

    def __getattr__(self, item):
        return getattr(self._ws, item)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            message = await self._ws.receive()
        except WebSocketDisconnect as ex:
            raise StopAsyncIteration from ex

        msg_type = message.get("type")

        if msg_type == "websocket.disconnect":
            raise StopAsyncIteration

        if msg_type != "websocket.receive":
            raise StopAsyncIteration

        if message.get("text") is not None:
            return message["text"]

        if message.get("bytes") is not None:
            return message["bytes"]

        return ""

    async def send(self, data):
        if isinstance(data, bytes):
            await self._ws.send_bytes(data)
        else:
            await self._ws.send_text(data)

    async def close(self, code: int = 1000, reason: str = ""):
        await self._ws.close(code=code, reason=reason)


class SpeakerEngineProcess:
    def __init__(self):
        self.session_manager = None
        self.max_sessions = (16)
        self._admission_lock = asyncio.Lock()
        self._active_sessions = 0
        ray.init(
            ignore_reinit_error=True,
            resources={"NPU": 1},
            logging_level=logging.INFO,
            log_to_driver=True,
            local_mode=False,
            address="local",
            _node_ip_address="127.0.0.1",
            runtime_env={"env_vars": {"PYTHONPATH": f"{current_dir}:$PYTHONPATH"}},
        )

    async def _send_overload_and_close(
        self,
        websocket: WebSocket,
        code: int = 1013,
        reason: str = "System Overloaded",
    ):
        payload = {
            "code": 111,
            "des": "System Overloaded",
        }

        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"send overload message failed: {e}")

        try:
            await websocket.close(code=code, reason=reason)
        except Exception as e:
            logger.warning(f"close websocket after overload failed: {e}")

    # 必选，加载模型
    def load(self):
        logger.info("load model start~")
        self.session_manager = SessionManager()
        logger.info("load model end~")

    async def _try_acquire_slot(self) -> bool:
        async with self._admission_lock:
            if self._active_sessions >= self.max_sessions:
                return False
            self._active_sessions += 1
            return True

    async def _release_slot(self):
        async with self._admission_lock:
            if self._active_sessions > 0:
                self._active_sessions -= 1

    # 可选，建链时做一些初始化操作
    def init(self):
        logger.info(
            "init called, pid=%s, tid=%s",
            os.getpid(),
            threading.get_ident(),
        )

    # 可选，断链时做一些清理操作
    def destroy(self):
        logger.info(
            "destroy called, pid=%s, tid=%s",
            os.getpid(),
            threading.get_ident(),
        )

    # 必选，推理方法
    async def calc_ws(self, ws_conn: WsConnector, work_msg: WorkMessage):
        session = None
        slot_acquired = False
        request_id = None
        try:
            raw_websocket = ws_conn.ws

            # 优先检查资源，不足直接拒绝
            slot_acquired = await self._try_acquire_slot()
            if not slot_acquired:
                logger.warning("session slot exhausted, reject connection")
                await self._send_overload_and_close(raw_websocket)
                return

            websocket = AsyncIterableWebSocketAdapter(raw_websocket)

            initial_message = work_msg.text
            if initial_message is None:
                raise ValueError("initial message is None")

            payload = json.loads(initial_message)
            request_id = payload.get("messageId", "")
            ContextHelper.set(request_id=request_id)
            initial_request = InitialSpeakerRequest.model_validate(payload)

            session = await self.session_manager.create_session(websocket, initial_request)

            if session:
                logger.info("session create success, wait websocket closed")
                try:
                    await asyncio.wait_for(
                        session.closed_event.wait(),
                        timeout=18300
                    )
                except asyncio.TimeoutError:
                   pass
                logger.info("check websocket closed, start to remove session")
            else:
                logger.error(f"Session creation failed for {request_id}")

        except Exception as e:
            logger.error(f"Unhandled error in calc_ws for {request_id}: {e}", exc_info=True)

        finally:
            try:
                if session:
                    await self.session_manager.remove_session(session)
            finally:
                if slot_acquired:
                    await self._release_slot()
                ContextHelper.clear()


    # 可选，业务可以自定义健康检查的逻辑
    def health_check(self):
        healthy = self.session_manager is not None and 0 <= self._active_sessions <= self.max_sessions
        slots_left = self.max_sessions - self._active_sessions
        logger.info(
            f"health check: pid={os.getpid()} tid={threading.get_ident()} "
            f"healthy={healthy}, "
            f"active_sessions={self._active_sessions}/{self.max_sessions}, "
            f"slots_left={slots_left}/{self.max_sessions}"
        )
        return healthy


