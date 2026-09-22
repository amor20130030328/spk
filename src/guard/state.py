# coding=utf-8
"""防护状态管理"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .guard import GuardConfig, GuardMode, GuardVerdict, StreamingGuard


@dataclass
class GuardedStreamState:
    """轻量级防护状态，用于包装器模式"""

    trusted_text: str = ""
    hallucination_streak: int = 0
    guard: StreamingGuard = field(default_factory=StreamingGuard)
    vad_speech_hint: Optional[bool] = None
    speech_sec: Optional[float] = None

    def reset(self):
        """完全重置防护状态"""
        self.trusted_text = ""
        self.hallucination_streak = 0
        self.vad_speech_hint = None
        self.speech_sec = None
