# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import ray

from src.opus_pcm.opus_decode import OpusToPcmConverter
from src.logger.logger_adapter import logger
from itn.chinese.inverse_normalizer import InverseNormalizer


class ItnRayActor:
    """opus转pcm actor实例"""

    def __init__(self, config):
        self._normalizer = InverseNormalizer(
            cache_dir=config.itn_path,
            overwrite_cache=False,
            enable_standalone_number=True,
            enable_0_to_9=False,
            enable_million=False,
        )
        self.is_ready = True
        text = "中华人民共和国成立于一九四九年十月一日"
        text2 = self._normalizer.normalize(text)
        logger.info(f"ItnRayActor {ray.get_runtime_context().get_actor_id()} initialized.")
        logger.info(f"ItnRayActor {text} | {text2} initialized.")

    def normalize(self, text) -> bytes:
        
        return self._normalizer.normalize(text)

    def ready(self) -> bool:
        """一个简单的健康检查方法，用于确认 Actor 已初始化完毕。"""
        return self.is_ready


