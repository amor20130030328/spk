# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import numpy as np
import ray
from numpy.typing import NDArray
import os
from src.logger.logger_adapter import logger
from src.vpr.lite_base_model import BaseLiteModel
from src.vpr.feature.fileio import read_hyperyaml

class VprRayActor:
    """vad actor实例"""
    def __init__(self, config):
        logger.info(f"VprRayActor initialized.")
        self.vpr_base_model = BaseLiteModel(config.vpr_path)
        self.is_ready = True
        self.yaml_path = "/model-data/component/src/vpr/feature/train.yaml"
        self.feature_extractor, self.sample_rate, self.dur_range = read_hyperyaml(path=self.yaml_path)
        logger.info(f"VprRayActor vpr init successfully. {self.yaml_path}")


    def process_audio(
            self,
            segments,
            frame,
            vad_start: int
    ):
        return self.vpr_model.handle(segments, frame, vad_start)

    def ready(self) -> bool:
        """一个简单的健康检查方法，用于确认 Actor 已初始化完毕。"""
        return self.is_ready


