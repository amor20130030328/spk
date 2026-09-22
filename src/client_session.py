# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import asyncio
import json
import time
from dataclasses import asdict
from typing import Optional
import torch
import numpy as np
import ray
from src.speaker.speaker_main_process import SpeakerMainProcess
from src.speaker.asr_main_process import AsrMainProcess
from src.actors.opus_to_pcm_actor import OpusToPcmConverterRayActor
from src.actors.silero_vad_actor import VadRayActor
from src.actors.itn_actor import ItnRayActor
from src.actors.vpr_actor import VprRayActor
from src.definitions import InitialSpeakerRequest
from src.definitions import (
    MonitorMetrics, AsrResult, VadResult, ExtInfo,SpeakerLongResponse, SpeakerLog
)
from src.logger.logger_adapter import get_logger
from src.exception.error_codes import ErrorCode, ServiceError
from src.exception.exception_handlers import handle_exceptions
from src.configs.config import config
from src.monitor.monitor_manager import MonitorManager
from src.guard.guard import GuardConfig,GuardContext, GuardMode, GuardVerdict, StreamingGuard
from src.guard.state import GuardedStreamState


class ClientSession:
    """封装单个WebSocket连接的所有逻辑和资源"""

    def __init__(
            self,
            initial_request: InitialSpeakerRequest,
            websocket,
            opus_actor: Optional[OpusToPcmConverterRayActor] = None,
            vad_actor: Optional[VadRayActor] = None,
            vpr_actor: Optional[VprRayActor] = None,
            itn_actor: Optional[ItnRayActor] = None,
            session_manager = None
    ):
        self.speaker_main_process = SpeakerMainProcess(initial_request, vpr_actor, itn_actor, session_manager, self)
        self.asr_main_process = AsrMainProcess(initial_request,itn_actor, session_manager)
        self.logger = get_logger({"sessionId": initial_request.sessionId})
        self.request = initial_request
        self.sessionId = initial_request.sessionId  # ClientSession均使用session_id来指代messageId
        self.websocket = websocket

        self.opus_actor = opus_actor
        self.vad_actor = vad_actor
        self.vpr_actor = vpr_actor
        self.itn_actor = itn_actor

        # audio 状态信息
        self.wave_buffer = np.array([], dtype=np.int16)
        self.total_wave_buffer = np.array([], dtype=np.int16)
        self.handle_time_line = 0
        self.offSize = 0  # 全局时间戳，ms
        self.tail_off = 0

        self.r_sn = 0
        self.speed_end = False
        self._wave_buffer_lock = asyncio.Lock()
        # 移除全局 _process_lock，改用细粒度锁控制
        self._result_send_queue = asyncio.Queue()

        # 状态管理
        self._tasks: list[asyncio.Task] = []
        self.is_running = False
        self.is_asr_status = False
        self.break_point = 0
        self.chunk_start_sample = 0

        # 新增：监控指标
        self._metrics = MonitorMetrics()
        self._monitor_task = None
        self.session_manager = session_manager
        self.monitor_manager = MonitorManager(initial_request.sessionId, initial_request.deviceId, self.logger)
        self.latest_speaker_log : SpeakerLog = None
        self.logger.info(f"deviceId = {self.request.deviceId}")
        self.closed_event = asyncio.Event()
        guard_config = GuardConfig(
            repeat_threshold=config.GUARD_REPEAT_THRESHOLD,
            max_chars_per_sec=config.GUARD_MAX_CHARS_PER_SEC,
            max_delta_chars=config.GUARD_MAX_DELTA_CHARS,
            singleton_ratio=config.GUARD_SINGLETON_RATIO,
            singleton_min_len=config.GUARD_SINGLETON_MIN_LEN,
            hallucination_streak_reset=config.GUARD_HALLUCINATION_STREAK_RESET,
        )
        self.guard = StreamingGuard(guard_config)
        self.guard_state = GuardedStreamState(guard=self.guard)

    def get_opus_actor(self):
        return self.opus_actor

    def get_vad_actor(self):
        return self.vad_actor

    def get_vpr_actor(self):
        return self.vpr_actor

    def get_itn_actor(self):
        return self.itn_actor

    def _apply_guard(self, candidate_text: str, audio_sec: float, mode: GuardMode) -> GuardVerdict:
        """应用 Guard 幻觉防护评估"""
        ctx = GuardContext(
            candidate_text=candidate_text,
            trusted_text=self.guard_state.trusted_text,
            audio_sec=audio_sec,
            raw_text=candidate_text,
            mode=mode,
            vad_speech_hint=self.guard_state.vad_speech_hint,
            speech_sec=self.guard_state.speech_sec,
        )
        verdict = self.guard.evaluate(ctx)

        if verdict.accept:
            self.guard_state.trusted_text = candidate_text
            self.guard_state.hallucination_streak = 0
        else:
            self.guard_state.hallucination_streak += 1
            if self.guard_state.hallucination_streak >= self.guard.config.hallucination_streak_reset:
                self.guard_state.trusted_text = ""
                self.guard_state.hallucination_streak = 0

        return verdict

    @handle_exceptions
    async def start(self):
        """启动会话的所有后台任务"""
        if self.is_running:
            return
        self.is_running = True
        self.logger.info("Starting client session.")
        self.guard_state.reset()
        try:
            self._tasks = [
                asyncio.create_task(self._receive_handler()),
                asyncio.create_task(self._processing_pipeline()),
                asyncio.create_task(self._send_handler()),
            ]
            # 监控任务
            self._monitor_task = asyncio.create_task(self._monitor_session(), name=f"monitor_{self.sessionId}")
            self.logger.info(f"Session {self.sessionId} started with {len(self._tasks)} tasks")
        except Exception as e:
            self.is_running = False
            raise ServiceError(
                error_code=ErrorCode.SYSTEM_ERROR,
                error_msg="Failed to start session",
                details=str(e),
            ) from e

    async def stop(self):
        """停止并清理会话的所有任务"""
        if not self.is_running:
            return
        self.is_running = False
        self.logger.info("Forcefully stopping client session.")
        self.logger.info(f"Session({self.sessionId}) metrics: {str(self._metrics)}")
        self.guard_state.reset()
        # 1. 取消所有任务
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # 2. 分别等待每个任务，处理取消异常
        results = []
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                # 任务正常被取消
                self.logger.debug(f"Task {task.get_name()} cancelled")
                pass
            except Exception as e:
                # 记录非取消异常
                self.logger.warning(f"Task {task.get_name()} raised exception: {e}")
                results.append(e)

        # 3. 处理监控任务
        if hasattr(self, '_monitor_task') and self._monitor_task:
            if not self._monitor_task.done():
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.warning(f"Monitor task error: {e}")

        # 4. 关闭 WebSocket
        try:
            await self.websocket.close(code=ErrorCode.FORCE_SHUTDOWN, reason="Session stopped")
        except Exception as e:
            # 连接可能已经由客户端关闭，这里记录一下即可
            self.logger.warning(f"Error closing WebSocket during forceful stop: {e}")

        self.logger.info("Client session forcefully stopped.")

    @handle_exceptions
    async def _receive_handler(self):
        self.logger.info("Task: _receive_handler begin")
        _receive_chunk_index = 0
        try:
            async for message in self.websocket:
                self._metrics.lastActivity = time.time()
                if not self.is_running:
                    break

                if isinstance(message, str):
                    self._metrics.messagesReceived += 1
                    self.logger.info(f"receive message: {message}")

                    if message == "--end--":
                        self.speed_end = True
                    else:
                        self.logger.info(f"Unknown message type: {message}")

                if isinstance(message, bytes):
                    start_time = time.time()
                    self._metrics.audioReceived += 1
                    try:
                        if self.opus_actor:
                            message = self.opus_actor.decode_opus(message, False)
                            # 根据配置决定是否保存音频文件
                            if config.save_audio_to_file:
                                self.monitor_manager.save_bytes_to_file(message, "pcm")
                            self.logger.debug(f"opus decode cost: {time.time() - start_time:.3f},  {len(message)}")
                        data = np.frombuffer(message, dtype=np.int16)
                        async with self._wave_buffer_lock:
                            self.wave_buffer = np.concatenate((self.wave_buffer, data))
                            self.total_wave_buffer = np.concatenate((self.total_wave_buffer, data))
                        if _receive_chunk_index % 50 == 0:
                            self.logger.debug(f"Received audio chunk {_receive_chunk_index}")
                        _receive_chunk_index += 1
                    except asyncio.TimeoutError:
                        self.logger.error(f"Opus conversion timeout when deal chunk {_receive_chunk_index}.")
                        self._metrics.errors += 1
                    except Exception as e:
                        self.logger.error(f"Error processing audio: {e}")
                        self._metrics.errors += 1
        except asyncio.CancelledError:
            self.logger.info("Receive handler cancelled.")
        except Exception as e:
            self.logger.error(f"Error in receive handler: {e}", exc_info=True)
        finally:
            self.closed_event.set()

    async def _handler_speaker(self, vad_res : VadResult, asr_text : str):
        try:
            async with self._wave_buffer_lock:
                start = vad_res.start - self.handle_time_line
                end = (self.handle_time_line + len(self.total_wave_buffer)) if vad_res.isEnd else (vad_res.end - self.handle_time_line)
                frame = self.total_wave_buffer[int(start):] if vad_res.isEnd else self.total_wave_buffer[:int(end)]
                self.handle_time_line = vad_res.end
                self.total_wave_buffer = self.total_wave_buffer[int(end):]
            self.logger.info(f"_handler_speaker vad_res = {vad_res} asr_text = {asr_text} frame = {len(frame)}")

            # segment_rms = compute_rms(frame)
            # if segment_rms < config.ENERGY_THRESHOLD:
            #     self.logger.info(f"speaker.compute_rms  {segment_rms} ENERGY_THRESHOLD = {config.ENERGY_THRESHOLD}")
            #     self.latest_speaker_log = SpeakerLog(frame=frame, text="", start=vad_res.start,
            #                                          end=vad_res.end)
            #     return ""

            speaker_infos, spk_text = await self.speaker_main_process.run(frame, vad_res, asr_text)
            if speaker_infos:
                await self.websocket.send(speaker_infos)
                self.latest_speaker_log = SpeakerLog(frame=frame, text=spk_text, start=vad_res.start,
                                                     end=vad_res.end)
                self.logger.info(
                    f"self.latest_speaker_log vad_res = {self.latest_speaker_log.text} frame = {len(self.latest_speaker_log.frame)}")
        except Exception as e:
            self.logger.error(f"speaker recognize error {str(e)}")


    async def do_qwen_asr_infer(self, vad_res: VadResult, wave_data) -> str:
        """Qwen‑ASR 核心推理业务逻辑，类成员方法，直接访问self"""
        async with self._wave_buffer_lock:
            end = vad_res.end - self.handle_time_line
            frame = self.total_wave_buffer[:int(end)]
        prev = ""
        if self.latest_speaker_log:
            prev = self.latest_speaker_log.text
            if wave_data is not None:
                self.logger.info(f"_handler_qwen_asr.vad_res concatenate {len(frame)}  {len(wave_data)} {len(np.concatenate((wave_data, frame)))}")
                frame = np.concatenate((wave_data, frame))

            frame = np.concatenate((self.latest_speaker_log.frame, frame))
        result, text = await self.asr_main_process.run(frame, prev, vad_res)
        if wave_data is not None:
            self.logger.info(f"_handler_qwen_asr.vad_res  {vad_res} prev = {prev} text = {text} frame.len = {len(frame)}")
        if result and text:
            st = time.time()
            audio_sec = (vad_res.end - vad_res.start) / config.sample_rate
            verdict = self._apply_guard(text, audio_sec, GuardMode.PARTIAL)
            self.logger.info(f"Guard cost time={time.time() - st}")
            if not verdict.accept:
                self.logger.info(f"Guard拦截(partial): reason={verdict.reason}, text={text} time={time.time() - st}")
                return self.guard_state.trusted_text

            await self.websocket.send(result)
            return text
        return ""


    async def _handler_qwen_asr(self, vad_res : VadResult):
        try:
            text = await self.do_qwen_asr_infer(vad_res, None)
            return text
        except Exception as e:
            self.logger.error("_handler_qwen_asr fail", str(e))
        return ""

    @handle_exceptions
    async def _send_handler(self):
        """从结果队列中取出结果，发送给客户端"""
        self.logger.info("Task: message send handler begin")
        asr_text = ""
        try:
            while True:
                self.logger.info(f"{self.sessionId} self._result_send_queue.qsize() = {self._result_send_queue.qsize()}")
                vad_res = await self._result_send_queue.get()
                self.is_asr_status = True
                # 修改点 2：检查哨兵值
                if vad_res is None:
                    self.logger.info("Send handler received sentinel. Shutting down.")
                    self._result_send_queue.task_done()
                    break
                try:
                    self.logger.debug(f'转写vad_res: {vad_res.start / config.sample_rate} {vad_res.end / config.sample_rate}  {vad_res.isFinal}')
                    if vad_res.isFinal:
                        await self._handler_speaker(vad_res, asr_text)
                        #  *** 关键，如何在此基础上提供转写结果
                    else:
                        asr_text = await self._handler_qwen_asr(vad_res)
                    self.is_asr_status = False
                except Exception as e:
                    self.logger.error(f"Failed to process current send result: {e}")
                    self._metrics.errors += 1
                finally:
                    self._result_send_queue.task_done()
                    self.is_asr_status = False

        except asyncio.CancelledError:
            self.logger.info("Send handler cancelled.")
        except Exception as e:
            self.logger.error(f"Error in send handler: {e}", exc_info=True)
        finally:
            self.logger.info("Task: message send handler end.")



    async def _processing_pipeline(self):
        """核心处理流水线"""
        self.logger.info("Task: processing pipeline begin")
        is_speech = False  # 追踪当前是否处于说话状态
        self.chunk_start_sample = 0
        self.break_point = 0      # 当前VAD处理到的时间节点 (采样点)
        asr_break_point = 0  # 当前转写到的时间节点    (采样点)
        asr_window_size_ms = 0.5
        vad_index = 0

        self.vad_actor.init_vad_state()
        try:
            while not self.speed_end:
                try:
                    self.logger.info(f"Task1: processing pipeline begin vad_index = {vad_index}")
                    batch_vad_frame = None
                    async with self._wave_buffer_lock:
                        wave_length = len(self.wave_buffer)
                        batch_size = wave_length // config.FRAME_SIZE
                        if batch_size > 0:
                            batch_vad_frame = self.wave_buffer[:(batch_size * config.FRAME_SIZE)]

                    if batch_vad_frame is None:
                        await asyncio.sleep(config.WAIT_TIME)
                        continue

                    for i in range(batch_size):
                        start_sample = i * config.FRAME_SIZE
                        end_sample = start_sample + config.FRAME_SIZE
                        audio_chunk = batch_vad_frame[start_sample:end_sample]
                        audio_chunk = audio_chunk.astype(np.float32) / config.audio_normalization_factor
                        frame_tensor = torch.from_numpy(audio_chunk).float()
                        if len(frame_tensor) < config.EXPECTED_SIZE:
                            frame_tensor = torch.nn.functional.pad(frame_tensor, (0, config.EXPECTED_SIZE - len(frame_tensor)))
                        frame_tensor = frame_tensor.unsqueeze(0)  # 添加batch维度
                        vad_start, vad_end = self.vad_actor.process(frame_tensor)

                        if vad_start:
                            is_speech = True
                            self.guard_state.vad_speech_hint = True
                        elif vad_end:
                            is_speech = False
                            self.guard_state.vad_speech_hint = False

                        end_sample = (vad_index + 1) * config.FRAME_SIZE
                        current_duration = (end_sample - self.chunk_start_sample) / config.sample_rate
                        should_cut = False

                        # 1. 软切割：如果当前切片长度进入了弹性窗口 (>= 12秒)，且当前处于静音状态
                        if current_duration >= config.MIN_TARGET_DURATION and not is_speech:
                            should_cut = True
                        # 2. 硬切割：如果一直说话，达到了最大强制阈值 (>= 18秒)，强制切分
                        elif current_duration >= config.MAX_TARGET_DURATION:
                            should_cut = True
                            # 强制切分后，状态依然是说话中，下一个切片会直接继承说话状态
                        if should_cut:
                            await self._result_send_queue.put(VadResult(self.chunk_start_sample, end_sample, True, False))
                            self.chunk_start_sample = end_sample
                        vad_index += 1

                    self.break_point += (batch_size * config.FRAME_SIZE)
                    async with self._wave_buffer_lock:
                        self.wave_buffer = self.wave_buffer[(batch_size * config.FRAME_SIZE):]

                    if self.break_point - asr_break_point > (asr_window_size_ms * config.sample_rate) and not self.is_asr_status:
                        asr_break_point = self.break_point
                        await self._result_send_queue.put(VadResult(self.chunk_start_sample, asr_break_point, False, False))
                except Exception as e:
                    self.logger.error(f"Error processing chunk: {e}")
                    self._metrics.errors += 1
            await self.on_speech_end()
        except asyncio.CancelledError:
            self.logger.info("Processing pipeline cancelled.")
            raise
        except Exception as e:
            self.logger.error(f"Critical error in processing pipeline: {e}", exc_info=True)
            asyncio.create_task(self.stop())
        finally:
            self.logger.info("Task: processing pipeline end.")

    async def on_speech_end(self):
        """
        语音流结束时的兜底处理：
        1. 处理缓冲区剩余音频
        2. 发送最终结果
        3. 关闭任务
        """
        try:
            last_end = self.chunk_start_sample
            self.logger.info("==================== on_speech_end 开始收尾 ====================")
            # 1. 取出缓冲区剩余所有音频
            async with self._wave_buffer_lock:
                remaining_frame = self.wave_buffer.copy()
            last_end_s = last_end / config.sample_rate
            vad_end_s = (self.break_point + len(remaining_frame)) / config.sample_rate
            if vad_end_s - last_end_s > config.MIN_SEGMENT_DURATION:
                start_index = int(last_end)
                end_index = int(self.break_point + len(remaining_frame))
                await self._result_send_queue.put(VadResult(start_index, end_index, True, False))

            # 4. 给结果队列发送哨兵值，让 send_handler 安全退出
            await self._result_send_queue.put(None)
            # 5. 等待队列处理完
            await self._result_send_queue.join()
            self.logger.info("==================== on_speech_end 收尾完成 ====================")

        except Exception as e:
            self.logger.error(f"on_speech_end 异常: {e}", exc_info=True)

        finally:
            # 6. 最终关闭会话
            await self.stop()

    async def _monitor_session(self):
        """监控会话健康状态"""
        check_interval = 5  # 每5秒检查一次

        try:
            while self.is_running:
                await asyncio.sleep(check_interval)

                # 检查任务状态
                dead_tasks = [t for t in self._tasks if t.done()]
                if dead_tasks:
                    for task in dead_tasks:
                        if task.exception():
                            self.logger.warning(f"Task {task.get_name()} failed: {task.exception()}")

                # 检查活动状态
                inactive_time = time.time() - self._metrics.lastActivity
                if inactive_time > config.SESSION_TIMEOUT:
                    self.logger.warning(f"Session inactive for {inactive_time:.1f}s, shutting down")
                    await self.stop()
                    break

                # 记录统计信息
                self.logger.debug(
                    f"Session stats - Received: {self._metrics.messagesReceived}, "
                    f"Sent: {self._metrics.messagesSent}, Errors: {self._metrics.errors}"
                )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Monitor error: {e}")


