# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import ray

from src.opus_pcm.opus_decode import OpusToPcmConverter
from src.logger.logger_adapter import logger


class OpusToPcmConverterRayActor:
    """opus转pcm actor实例"""

    def __init__(self):
        self.opus_convertor = OpusToPcmConverter()
        self.is_ready = True
        logger.info(f"OpusToPcmConverterRayActor {ray.get_runtime_context().get_actor_id()} initialized.")

    def decode_opus(self, frame_data: bytes, is_short: bool) -> bytes:
        if is_short:
            return self.opus_convertor.decode_short_opus(frame_data)
        else:
            return self.opus_convertor.decode_long_opus(frame_data)

    def ready(self) -> bool:
        """一个简单的健康检查方法，用于确认 Actor 已初始化完毕。"""
        return self.is_ready

