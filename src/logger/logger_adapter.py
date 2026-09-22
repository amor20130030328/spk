# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import logging
import os
from typing import Any, Tuple
from typing import Dict, Optional

from src.context.context_helper import ContextHelper


class LoggerAdapter(logging.LoggerAdapter):

    def __init__(self, run_logger, extra=None):
        super().__init__(run_logger, extra or {})

    def process(self, msg: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        # 1. 优先从直接传参中获取 (logger.info(..., request_id="xxx"))
        req_id = kwargs.pop("request_id", None)

        # 2. 如果没传，去 ContextHelper 里拿 (最常用)
        if not req_id:
            req_id = ContextHelper.get("request_id")

        # 3. 如果还是没有，尝试从 extra 初始化参数里拿
        if not req_id and self.extra:
            req_id = self.extra.get("request_id")

        # 4.直接修改 msg 字符串
        # 原始 msg: "end state_transition..."
        # 修改后: "[user_123] end state_transition..."
        if req_id:
            msg = f"[{req_id}] {str(msg)}"

        return msg, kwargs


def get_logger(extra: Optional[Dict[str, str]] = None):
    from common.util import logutil

    _raw_logger = logutil.logger_run
    new_logger = LoggerAdapter(_raw_logger, extra)
    return new_logger


logger = get_logger({})
logger_level = os.environ.get("MEP_FRAMEWORK_RUN_LOG_LEVEL", 'INFO')
if logger_level == 'DEBUG':
    logger.setLevel(logging.DEBUG)
