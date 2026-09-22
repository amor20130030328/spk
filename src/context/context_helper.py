# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

import contextvars


class ContextHelper:
    """上下文变量帮助类

    Attributes:
        __stream_context: 上下文变量
    """

    __stream_context = contextvars.ContextVar('stream_context', default={})

    @staticmethod
    def get(key: str):
        """获取上下文变量

        Args:
            key: 上下文变量名称
        Returns:
            上下文变量值
        """
        context_dict = ContextHelper.__stream_context.get()
        return context_dict.get(key)

    @staticmethod
    def set(**kwargs):
        """设置上下文变量

        Args:
            kwargs: 上下文变量键值对
        """
        ctx_dict = ContextHelper.__stream_context.get()

        # 2. 浅拷贝（为了不影响父级或其他协程）
        new_dict = ctx_dict.copy()

        # 3. 增量更新（把新参数加进去）
        new_dict.update(kwargs)

        # 4. 重新赋值
        ContextHelper.__stream_context.set(new_dict)

    @staticmethod
    def clear():
        """清除上下文变量"""

        ContextHelper.__stream_context.set({})
