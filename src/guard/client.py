# coding=utf-8
"""包装器实现 - 包装 ASR 客户端，自动应用防护"""

from __future__ import annotations
from typing import Optional
from .guard import GuardContext, GuardMode, GuardVerdict
from .state import GuardedStreamState


class GuardedAsrClient:
    """包装 ASR 客户端，在 transcribe 调用后自动应用防护"""

    def __init__(self, underlying_client, config=None):
        self.underlying = underlying_client
        self.state = GuardedStreamState(guard=None)
        from .guard import StreamingGuard
        self.state.guard = StreamingGuard(config)

    def transcribe(
        self,
        wav,
        prev_src="",
        src_context="",
        tag="",
        duration_sec=0.0,
        enable_fa: bool = False,
        mode: Optional[GuardMode] = None,
    ) -> dict:
        """调用底层 transcribe，然后应用防护"""
        result = self.underlying.transcribe(
            wav, prev_src=prev_src, src_context=src_context,
            tag=tag, duration_sec=duration_sec, enable_fa=enable_fa,
        )


        if mode is None:
            mode = GuardMode.FINAL if enable_fa else GuardMode.PARTIAL

        verdict = self._apply_guard(
            result.get("processed_text", ""),
            result.get("raw_text", ""),
            duration_sec,
            mode,
        )

        result["guard_verdict"] = {
            "accept": verdict.accept,
            "reason": verdict.reason,
            "severity": verdict.severity,
            "display_ok": verdict.display_ok,
        }

        if not verdict.accept:
            rolled_back = self.state.trusted_text
            result["processed_text"] = rolled_back
            result["raw_text"] = rolled_back
            result["timestamps"] = []
            result["fa_meta"] = {}

        return result

    def _apply_guard(
        self,
        candidate_text: str,
        raw_text: str,
        audio_sec: float,
        mode: GuardMode,
    ) -> GuardVerdict:
        ctx = GuardContext(
            candidate_text=candidate_text,
            trusted_text=self.state.trusted_text,
            audio_sec=audio_sec,
            raw_text=raw_text,
            mode=mode,
            vad_speech_hint=self.state.vad_speech_hint,
            speech_sec=self.state.speech_sec,
        )
        verdict = self.state.guard.evaluate(ctx)

        if verdict.accept:
            self.state.trusted_text = candidate_text
            self.state.hallucination_streak = 0
            return verdict

        self.state.hallucination_streak += 1

        if self.state.hallucination_streak >= self.state.guard.config.hallucination_streak_reset:
            self.state.trusted_text = ""
            self.state.hallucination_streak = 0

        return verdict

    def reset(self):
        """重置防护状态"""
        self.state.reset()
