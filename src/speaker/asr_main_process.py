import numpy as np
from src.logger.logger_adapter import get_logger
from src.utils.http_util import request_qwen3_asr
from src.definitions import AsrResult
import time
from src.definitions import SpeakerLongResponse, InitialSpeakerRequest, ExtInfo, TimeInfo
from src.configs.config import config
import json
from dataclasses import asdict
from src.utils.meeting_processor import processor
from src.utils.audio_utils import compute_rms

class AsrMainProcess:

    def __init__(self, request : InitialSpeakerRequest, itn_actor, session_manager):
        self.session_id = request.sessionId
        self.deviceId = request.deviceId
        self.itn_actor = itn_actor
        self.session_manager = session_manager
        self.logger = get_logger({"session_id": self.session_id})


    def __make_result(self, result, vad_res, infer_time):

        if self.deviceId != config.debug_device_Id:
            return json.dumps(asdict(result), ensure_ascii=False)
        else:
            ext_info = ExtInfo(activate_session=self.session_manager.get_all_active_count())
            time_info = TimeInfo(g_bg=vad_res.start / config.sample_rate,
                                 g_ed=vad_res.end / config.sample_rate,
                                 infer_time=infer_time)
            return json.dumps(asdict(SpeakerLongResponse(sessionId=self.session_id,
                                       deviceId=self.deviceId,
                                       vendor= config.asr_model,
                                       result=result,
                                       time_info=time_info,
                                       ext_info=ext_info)), ensure_ascii=False)


    async def run(self, frame, prev : str, vad_res):
        """异步版本的 ASR 处理"""
        try:
            overall_start = time.time()

            # 1. 数据预处理
            preprocess_start = time.time()
            frame = frame.astype(np.float32) / config.audio_normalization_factor
            preprocess_time = time.time() - preprocess_start

            # 2. HTTP 请求（包含等待信号量时间）
            http_start = time.time()
            semaphore = self.session_manager.get_http_semaphore()
            result = await request_qwen3_asr(self.session_id, frame, prev, semaphore=semaphore)
            http_time = time.time() - http_start

            # 3. 后处理
            if result and "sentence" in result:
                postprocess_start = time.time()
                raw_text = result['sentence']
                processed_text = processor.process(raw_text, text_lang="zh")
                processed_text = self.itn_actor.normalize(processed_text)
                postprocess_time = time.time() - postprocess_start

                overall_time = time.time() - overall_start

                # 详细耗时日志
                self.logger.info(
                    f"[ASR_TIMING] total={overall_time:.3f}s | "
                    f"preprocess={preprocess_time:.3f}s | "
                    f"http={http_time:.3f}s | "
                    f"postprocess={postprocess_time:.3f}s | "
                    f"text_len={len(processed_text)}"
                )

                return self.__make_result(AsrResult(text=processed_text, isFinish=False, retType="partial"), vad_res, overall_time), processed_text
        except Exception as e:
            self.logger.error(f"asr recognize fail {str(e)}")
        return None, ""



