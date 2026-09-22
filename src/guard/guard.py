# coding=utf-8
"""独立防护实现 - 不依赖任何外部qwen3asr文件"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple


class GuardMode(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"


@dataclass
class GuardConfig:
    repeat_threshold: int = 8
    max_chars_per_sec: float = 10.0
    max_delta_chars: int = 25
    singleton_ratio: float = 0.7
    singleton_min_len: int = 6
    hallucination_streak_reset: int = 2
    known_hallucination_patterns: Tuple[str, ...] = (
        "you are a helpful assistant",
        "你是一个乐于助人的助手",
        "helpful assistant",
        "<asr_text>",
        "language none",
    )
    book_intro_min_len: int = 2
    book_intro_keywords: Tuple[str, ...] = (
        "创作", "作者", "作家", "导演", "主演", "出版", "首次出版", "首次发表",
        "长篇小说", "短篇小说", "散文", "诗集", "编剧", "制作", "发表",
        "代表作", "改编", "原著", "原名",
        "法国", "英国", "美国", "日本", "德国", "俄国", "中国",
    )


@dataclass
class GuardVerdict:
    accept: bool
    reason: str = ""
    severity: str = "hard"
    display_ok: bool = True


@dataclass
class GuardContext:
    candidate_text: str
    trusted_text: str
    audio_sec: float
    raw_text: str = ""
    mode: GuardMode = GuardMode.PARTIAL
    vad_speech_hint: Optional[bool] = None
    speech_sec: Optional[float] = None


def _dominant_char_ratio(text: str) -> Tuple[str, float]:
    if not text:
        return "", 0.0
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isspace():
            continue
        counts[ch] = counts.get(ch, 0) + 1
    if not counts:
        return "", 0.0
    top_char = max(counts, key=counts.get)
    visible = sum(counts.values())
    return top_char, counts[top_char] / visible


def _has_residual_repetition(text: str, threshold: int) -> bool:
    if not text:
        return False

    streak = 1
    prev = None
    for ch in text:
        if ch == prev:
            streak += 1
            if streak >= threshold:
                return True
        else:
            prev = ch
            streak = 1

    compact = re.sub(r"\s+", "", text)
    n = len(compact)
    if n < threshold * 2:
        return False
    max_pat = min(12, n // threshold)
    for size in range(1, max_pat + 1):
        for start in range(0, n - size * threshold + 1):
            pattern = compact[start : start + size]
            if not pattern:
                continue
            end = start + size * threshold
            if compact[start:end] == pattern * threshold:
                return True
    return False


def _matches_known_hallucination(text: str, patterns: Sequence[str]) -> bool:
    lower = text.lower()
    return any(p in lower for p in patterns)


def _matches_book_intro_hallucination(
    text: str,
    min_len: int,
    keywords: Sequence[str],
    book_names: Sequence[str] = (),
) -> bool:
    if not text or len(text) < min_len:
        return False
    if text.startswith("《"):
        return True
    if "》" in text and any(k in text for k in keywords):
        return True
    return False


class StreamingGuard:
    def __init__(self, config: Optional[GuardConfig] = None):
        self.config = config or GuardConfig()

    def evaluate(self, ctx: GuardContext) -> GuardVerdict:
        text = (ctx.candidate_text or "").strip()
        if not text:
            return GuardVerdict(accept=True, reason="empty", severity="soft")

        cfg = self.config

        if _matches_known_hallucination(text, cfg.known_hallucination_patterns):
            return self._reject("known_hallucination", ctx)

        raw_text = (ctx.raw_text or "").strip()
        if _matches_book_intro_hallucination(
            raw_text, cfg.book_intro_min_len, cfg.book_intro_keywords
        ):
            return self._reject("book_intro_hallucination", ctx)
        if _matches_book_intro_hallucination(
            text, cfg.book_intro_min_len, cfg.book_intro_keywords
        ):
            return self._reject("book_intro_hallucination", ctx)

        if _has_residual_repetition(text, cfg.repeat_threshold):
            return self._reject("repeat_pattern", ctx)

        _, ratio = _dominant_char_ratio(text)
        visible_len = len(re.sub(r"\s+", "", text))
        if visible_len >= cfg.singleton_min_len and ratio >= cfg.singleton_ratio:
            return self._reject("singleton_ratio", ctx)

        if ctx.audio_sec > 0:
            chars_per_sec = visible_len / ctx.audio_sec
            if chars_per_sec > cfg.max_chars_per_sec:
                return self._reject("char_rate", ctx)

        trusted_visible_len = len(re.sub(r"\s+", "", (ctx.trusted_text or "")))
        delta = max(0, visible_len - trusted_visible_len)
        if delta > cfg.max_delta_chars:
            return self._reject("delta_explosion", ctx)

        if ctx.mode == GuardMode.FINAL and ctx.speech_sec is not None:
            if ctx.speech_sec > 0 and visible_len / ctx.speech_sec > cfg.max_chars_per_sec * 1.5:
                return self._reject("final_char_rate", ctx)
            if ctx.speech_sec < 0.3 and visible_len >= cfg.singleton_min_len:
                return self._reject("final_speech_too_short", ctx)

        display_ok = True
        if ctx.vad_speech_hint is False:
            display_ok = False

        return GuardVerdict(accept=True, reason="ok", severity="soft", display_ok=display_ok)

    def _reject(self, reason: str, ctx: GuardContext) -> GuardVerdict:
        return GuardVerdict(accept=False, reason=reason, severity="hard", display_ok=False)
