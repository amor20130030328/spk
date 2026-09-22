import numpy as np
from src.logger.logger_adapter import get_logger
from src.utils.http_util import request_qwen3_asr
from src.definitions import AsrResult
import time
from src.definitions import SpeakerLongResponse, InitialSpeakerRequest, ExtInfo, TimeInfo
from src.configs.config import config
import json
from dataclasses import asdict
from src.utils.omni_postprocess import build_omni_response
from src.utils.meeting_processor import processor

class FaMainProcess:

    def __init__(self, request : InitialSpeakerRequest, itn_actor, session_manager):
        self.session_id = request.sessionId
        self.deviceId = request.deviceId
        self.itn_actor = itn_actor
        self.session_manager = session_manager
        self.logger = get_logger({"session_id": self.session_id})



    async def run(self, frame, vad_res):
        """异步版本的 FA 处理"""
        try:
            start_time = time.time()
            #frame = frame.astype(np.float32) / config.audio_normalization_factor
            semaphore = self.session_manager.get_http_semaphore()
            result = await request_qwen3_asr(self.session_id, data=frame, prev="", enable_fa=True, semaphore=semaphore)
            infer_time = round(time.time() - start_time, 3)

            if result and "sentence" in result and "timestamps" in result:
                text = self.itn_actor.normalize(result['sentence'])
                timestamps = result['timestamps']
                audio_sec = (vad_res.end - vad_res.start) / config.sample_rate
                vad_start_sec = round(vad_res.start / config.sample_rate, 3)
                omni_part = build_omni_response(
                    raw_text=text,
                    timestamps=timestamps,
                    audio_sec=audio_sec,
                    vad_start_sec=vad_start_sec,
                    text_processor=processor,
                    segment_index_offset=1,
                    infer_time=infer_time
                )

                return omni_part
        except Exception as e:
            self.logger.error(f"fa recognize fail {str(e)}")
        return None



