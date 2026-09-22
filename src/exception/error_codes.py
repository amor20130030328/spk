# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import asyncio
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class ErrorCode(IntEnum):
    """统一错误码定义"""
    SUCCESS = 0  # 成功

    # 系统级错误 1000-1999
    SYSTEM_ERROR = 1000  # 系统内部错误
    SERVICE_UNAVAILABLE = 1002  # 服务不可用
    FORCE_SHUTDOWN = 1011  # 强制关闭连接

    # 会话级错误 3000-3999
    SESSION_NOT_FOUND = 3001  # 会话不存在
    SESSION_CONFLICT = 3003  # 会话冲突

    OCCUPY_CHANNEL_FAILED = 4028  # S2TT engine occupy channel failed


@dataclass
class ServiceError(Exception):
    """服务异常基类"""
    error_code: ErrorCode
    error_msg: str
    details: Optional[str] = None
    retryable: bool = False

    def __post_init__(self):
        super().__init__(self.error_msg)

    @classmethod
    def from_exception(cls, error_code: ErrorCode, exception: Exception):
        """从现有异常创建"""
        return cls(
            error_code=error_code,
            error_msg=str(exception),
            details=repr(exception),
            retryable=isinstance(exception, (asyncio.TimeoutError, ConnectionError))
        )

    def to_dict(self):
        """转换为字典格式"""
        return {
            "code": self.error_code.value,
            "message": self.error_msg,
            "details": self.details,
            "retryable": self.retryable
        }
