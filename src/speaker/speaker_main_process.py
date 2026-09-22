import numpy as np
import time
from src.logger.logger_adapter import get_logger
from src.vpr.vpr import Vpr
import asyncio
from src.definitions import SpeakerLongResponse, InitialSpeakerRequest, ExtInfo, TimeInfo, VadResult
from src.utils.http_util import request_speaker_omni
from src.configs.config import config
from src.speaker.fa_main_process import FaMainProcess
import base64
import json
from dataclasses import asdict
from typing import Optional


class SpeakerMainProcess:

    def __init__(self, request : InitialSpeakerRequest, vpr_actor, itn_actor, session_manager, client_session):
        self.session_id = request.sessionId
        self.deviceId = request.deviceId
        self.logger = get_logger({"session_id": request.sessionId})
        self.itn_actor = itn_actor
        self.vpr_model = Vpr(vpr_actor, itn_actor, self.logger)
        self.fa_service = FaMainProcess(request , itn_actor, session_manager)
        self.session_manager  = session_manager
        self.client_session  = client_session
        self.logger.info(
            f"Initializing speaker main process = sessionId={self.session_id} deviceId={self.deviceId}")



    def __make_result(self, result, vad_res, omni_result, is_omni : bool):

        self.logger.info(
            f"speaker_infos start ")
        sentence = omni_result["sentence"]
        segments = omni_result["segments"]
        infer_time = omni_result["infer_time"]

        self.logger.info(
            f"speaker_infos infer_time={infer_time} ")

        if self.deviceId != config.debug_device_Id:
            self.logger.info(f"run_sync.omni speaker r2_ok1 = {is_omni} __make_result. {result}")
            return json.dumps(result, ensure_ascii=False)
        else:
            ext_info = ExtInfo(activate_session=self.session_manager.get_all_active_count())
            time_info = TimeInfo(g_bg=vad_res.start / config.sample_rate,
                                 g_ed=vad_res.end / config.sample_rate,
                                 infer_time=infer_time)
            vendor = config.omni_model if is_omni else config.fa_model
            speakerResult =  SpeakerLongResponse(action="final",
                                       sessionId=self.session_id,
                                       deviceId=self.deviceId,
                                       vendor= vendor,
                                       result=result,
                                       time_info=time_info,
                                       omni={
                                           "sentence": sentence,
                                           "segments": segments
                                       },
                                       ext_info=ext_info)
            return json.dumps(asdict(speakerResult), ensure_ascii=False)



    async def run(self, wave_data, vad_res, asr):
        overall_start = time.time()
        stop_asr_event: Optional[asyncio.Event] = None
        bg_asr_task: Optional[asyncio.Task] = None
        try:
            timeout_r2 = 3
            timeout_r1 = 3
            MAX_TASK_ASR_RUN = 3

            # 准备阶段
            prepare_start = time.time()
            stop_asr_event = asyncio.Event()

            async def loop_task_asr(frame):
                run_count = 0
                while run_count < MAX_TASK_ASR_RUN and not stop_asr_event.is_set():
                    try:
                        vad = VadResult(vad_res.start, self.client_session.break_point, False, False)
                        await self.client_session.do_qwen_asr_infer(vad, frame)
                        run_count += 1
                        self.logger.debug(f"run task_asr 已执行 {run_count}/{MAX_TASK_ASR_RUN} {vad}")
                    except Exception as e:
                        self.logger.warning(f"run loop_task_asr执行异常, count={run_count + 1}", exc_info=True)
                        run_count += 1
                    if run_count < MAX_TASK_ASR_RUN and not stop_asr_event.is_set():
                        await asyncio.sleep(0.02)

            bg_asr_task = asyncio.create_task(loop_task_asr(wave_data), name="bg_loop_task_asr")
            wave_data = wave_data.astype(np.float32) / config.audio_normalization_factor
            wave_duration = wave_data.shape[0] * 1.0 / config.sample_rate
            vad_start = str(vad_res.start / config.sample_rate)
            prepare_time = time.time() - prepare_start

            self.logger.info(f"run.Starting speaker regconized. {vad_res} {asr}  wave_duration = {wave_duration}")

            # 并发启动 r1(FA) 和 r2(Omni)
            r1_start = time.time()
            r2_start = time.time()

            r1_task = asyncio.create_task(
                 self.fa_service.run(wave_data, vad_res),
                 name="post_qwen3_asr_fa"
            )

            semaphore = self.session_manager.get_http_semaphore()
            r2_task = asyncio.create_task(
                request_speaker_omni(self.session_id, wave_data, asr, vad_start, semaphore=semaphore),
                name="post_qwen3_omni"
            )

            r2_result: Optional[object] = None
            r2_ok = False
            r2_time = 0.0

            # 2. 带超时获取 r2
            try:
                r2_result = await asyncio.wait_for(r2_task, timeout=timeout_r2)
                r2_time = time.time() - r2_start
                self.logger.info(f"[SPEAKER_R2_OMNI] completed in {r2_time:.3f}s")
                r2_ok = True
            except asyncio.TimeoutError:
                r2_time = time.time() - r2_start
                self.logger.warning(f"[SPEAKER_R2_OMNI] timeout after {r2_time:.3f}s")
            except Exception as e:
                r2_time = time.time() - r2_start
                self.logger.error(f"[SPEAKER_R2_OMNI] failed after {r2_time:.3f}s: {e}")

            selected: Optional[object] = None
            # 3. 优先走 r2
            if r2_ok and r2_result is not None:
                r2_str = str(r2_result).strip()
                if r2_str:
                    selected = r2_result
                    self.logger.info("run.优先使用 r2(post_omni2) 有效结果")
                    # 无用任务直接取消，释放资源
                    r1_task.cancel()
                else:
                    self.logger.warning("run.r2(post_omni2) 结果为空，降级使用 r1")
            else:
                self.logger.info("run.r2 不可用，降级等待 r1(post_omni1)")

            # 4. r2 不可用，处理 r1
            r1_time = 0.0
            if selected is None:
                try:
                    r1_result = await asyncio.wait_for(r1_task, timeout=timeout_r1)
                    r1_time = time.time() - r1_start
                    self.logger.info(f"[SPEAKER_R1_FA] completed in {r1_time:.3f}s")
                    if r1_result is not None and str(r1_result).strip():
                        selected = r1_result
                        self.logger.info("run.成功使用 r1(post_omni1) 降级结果")
                        self.logger.debug("run.成功使用 r1(post_omni1) 降级结果"+ str(selected))
                    else:
                        self.logger.warning("run.r1(post_omni1) 结果为空，无可用结果")
                except asyncio.TimeoutError:
                    r1_time = time.time() - r1_start
                    self.logger.error(f"[SPEAKER_R1_FA] timeout after {r1_time:.3f}s")
                except asyncio.CancelledError:
                    self.logger.warning("run.r1(post_omni1) 任务已被取消")
                except Exception as e:
                    r1_time = time.time() - r1_start
                    self.logger.error(f"[SPEAKER_R1_FA] failed after {r1_time:.3f}s: {e}")

            if selected is None or "segments" not in selected:
                return None, ""

            omni_result = selected
            self.logger.info(f"run.omni speaker omni_result. {omni_result}")

            # VPR 处理
            vpr_start = time.time()
            segments = omni_result["segments"]
            sentence = omni_result["sentence"]
            vad_start = float(vad_start)
            speaker_infos = self.vpr_model.handle(segments, wave_data, vad_start)
            vpr_time = time.time() - vpr_start
            self.logger.info(f"speaker_infos speaker_infos={speaker_infos}")

            wordsSegs = [w for seg in segments for w in seg['wordsSeg']]

            # ITN 处理
            itn_start = time.time()
            final_text = ""
            for speaker_info in speaker_infos:
                final_text = f"{final_text}{speaker_info['word']}"
            final_text = self.itn_actor.normalize(final_text)
            itn_time = time.time() - itn_start

            if len(speaker_infos) == 0:
                return None, ""

            result = {
                "speakerInfo": speaker_infos,
                "isFinish": "true",
                "words": "",
                "text": final_text,
                "wordsSeg" : wordsSegs,
                "isSpeaker": "true",
                "isLast": vad_res.isEnd,
                "bg": speaker_infos[0]["vadInfo"]["start_of_speech"] * 10,
                "ed": speaker_infos[-1]["vadInfo"]["end_of_speech"] * 10,
                "retType": "final"
            }

            # 总耗时统计
            overall_time = time.time() - overall_start
            self.logger.info(
                f"[SPEAKER_TIMING] total={overall_time:.3f}s | "
                f"prepare={prepare_time:.3f}s | "
                f"r2_omni={r2_time:.3f}s | "
                f"r1_fa={r1_time:.3f}s | "
                f"vpr={vpr_time:.3f}s | "
                f"itn={itn_time:.3f}s | "
                f"selected={'r2' if r2_ok else 'r1'}"
            )

            self.logger.info(f"speaker_infos.result {result}")
            result = self.__make_result(result, vad_res, omni_result, is_omni=r2_ok)
            self.logger.info(f"speaker_infos speaker__make_result_post. deviceId = {self.deviceId} {result}")
            return result, final_text
        except Exception as e:
            self.logger.error(f"speaker recognize fail {str(e)}")
            return None, ""
        finally:
            # ✅无论正常返回，还是异常崩溃，都要停止后台task_asr协程，杜绝协程泄漏
            if stop_asr_event is not None:
                stop_asr_event.set()
            if bg_asr_task is not None and not bg_asr_task.done():
                try:
                    await bg_asr_task
                except asyncio.CancelledError:
                    self.logger.debug("bg_asr_task已经取消")
                except Exception:
                    self.logger.warning("bg_asr_task 退出时发生异常", exc_info=True)






