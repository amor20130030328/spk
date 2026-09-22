# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import asyncio
import functools
from typing import Callable

from src.logger.logger_adapter import logger
from src.exception.error_codes import (
    ErrorCode,
    ServiceError,
)


def handle_exceptions(func: Callable):
    """异常处理装饰器"""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError as e:
            logger.error(f"Service CancelledError in {func.__name__}: {e}", exc_info=True)
            raise  # 传播取消错误
        except ServiceError as e:
            logger.error(f"Service error in {func.__name__}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise ServiceError.from_exception(ErrorCode.SYSTEM_ERROR, e)

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ServiceError as e:
            logger.error(f"Service error in {func.__name__}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise ServiceError.from_exception(ErrorCode.SYSTEM_ERROR, e)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
