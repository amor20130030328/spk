# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import ray
from src.logger.logger_adapter import logger
from src.silero_vad.utils_vad import OnnxWrapper, VADIterator
import time

#@ray.remote(num_cpus=1)
class VadRayActor:
    """vad actor实例"""

    def __init__(self, model_path: str):
        self.model = OnnxWrapper(model_path, force_onnx_cpu=True)
        self.vad_iterator = None
        self.init_vad_state()
        self.is_ready = True
        logger.info(f"VadRayActor {ray.get_runtime_context().get_actor_id()} initialized.")
    def process(
        self,
        audio
    ):
        st = time.time()
        result = self.vad_iterator(audio, return_seconds=True)
        start = None
        end = None
        if result:
            if 'start' in result:
                start = result['start']
            elif 'end' in result:
                end = result['end']
        et = time.time()
        logger.info(f"VadRayActor cost time = {et - st}.")
        return start, end

    def ready(self) -> bool:
        """一个简单的健康检查方法，用于确认 Actor 已初始化完毕。"""
        return self.is_ready


    def init_vad_state(self):
        self.vad_iterator = VADIterator(
            model=self.model,  # 复用模型，不重复加载
            threshold=0.5,
            sampling_rate=16000,
            min_silence_duration_ms=250,
            speech_pad_ms=30
        )
        logger.debug(f"VadRayActor {ray.get_runtime_context().get_actor_id()} reset completed.")

