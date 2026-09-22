"""FA 结果 → 标点断句 → OmniResponse（逻辑源自 sdr_eh_service_start.py）。"""
from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from src.logger.logger_adapter import logger

DP3 = Decimal("0.001")
MIN_DURATION = Decimal("0.35")
MICRO_STEP = 0.01
SPLIT_PUNCTS = {"。", "？", "！", ".", "?", "!"}


def segment_asr_by_punctuation(
    asr_data: dict[str, Any],
    max_audio_duration: float | None = None,
    vad_start_sec: float | None = None,
) -> list[dict[str, Any]]:
    """根据标点切分 FA 字级时间戳，含坍缩/重叠修复。"""
    full_text = asr_data.get("sentence", "")
    time_stamps_data = asr_data.get("segments", [])
    if isinstance(time_stamps_data, dict):
        items = time_stamps_data.get("items", [])
    else:
        items = time_stamps_data if isinstance(time_stamps_data, list) else []

    global_last_time = 0.0
    cleaned_items = []
    for it in items:
        start_t = float(it.get("start_time", 0.0))
        end_t = float(it.get("end_time", 0.0))
        char_text = it.get("text", "")

        if start_t == end_t:
            end_t = start_t + MICRO_STEP
        if start_t < global_last_time:
            start_t = global_last_time
            if end_t <= start_t:
                end_t = start_t + MICRO_STEP
        if end_t <= start_t:
            end_t = start_t + MICRO_STEP
        if max_audio_duration is not None:
            start_t = min(start_t, max_audio_duration)
            end_t = min(end_t, max_audio_duration)

        global_last_time = end_t
        cleaned_items.append({
            "text": char_text,
            "start_time": round(start_t, 3),
            "end_time": round(end_t, 3),
        })
    items = cleaned_items

    raw_segments = []
    cur_text = ""
    cur_start = None
    cur_end = None
    cur_words: list[dict] = []
    item_idx = 0
    item_char_idx = 0

    for char in full_text:
        cur_text += char
        if item_idx < len(items):
            current_item_text = items[item_idx].get("text", "")
            if item_char_idx < len(current_item_text) and char == current_item_text[item_char_idx]:
                if cur_start is None:
                    cur_start = items[item_idx]["start_time"]
                cur_end = items[item_idx]["end_time"]
                if item_char_idx == 0:
                    cur_words.append(items[item_idx])
                item_char_idx += 1
                if item_char_idx >= len(current_item_text):
                    item_idx += 1
                    item_char_idx = 0

        if char in SPLIT_PUNCTS:
            if cur_text.strip():
                raw_segments.append({
                    "text": cur_text.strip(),
                    "start_time": cur_start,
                    "end_time": cur_end,
                    "words": list(cur_words),
                })
            cur_text = ""
            cur_start = None
            cur_end = None
            cur_words = []

    if cur_text.strip():
        raw_segments.append({
            "text": cur_text.strip(),
            "start_time": cur_start,
            "end_time": cur_end,
            "words": list(cur_words),
        })

    valid_segments = []
    for seg in raw_segments:
        st = seg["start_time"]
        et = seg["end_time"]
        if st is None or et is None:
            continue
        if et <= st:
            et = st + 0.1
        seg["start_time"] = str(st)
        seg["end_time"] = str(et)
        words_list = seg.get("words", [])
        seg["chars"] = [
            {
                "index" : idx + 1,
                "word": w["text"],
                "bg": int(w["start_time"] * 100),
                "ed": int(w["end_time"] * 100),
                "vadBg": int(vad_start_sec * 100)
            }
            for idx, w in enumerate(words_list)
        ]
        valid_segments.append(seg)
    return valid_segments


def build_omni_response(
    raw_text: str,
    timestamps: list[dict],
    audio_sec: float,
    vad_start_sec: float,
    text_processor,
    src_lang: str = "zh",
    segment_index_offset: int = 0,
    infer_time: float | None = None,
) -> dict[str, Any]:
    """
    FA 输出 → 标点断句 + 文本清洗 + 时长过滤 → OmniResponse 片段列表。

    返回::
        {"sentence": str, "segments": list, "filtered_count": int}
    """

    asr_data = {"sentence": raw_text, "segments": timestamps}
    punct_segments = segment_asr_by_punctuation(asr_data, max_audio_duration=audio_sec, vad_start_sec=vad_start_sec)
    vad_start_d = Decimal(str(vad_start_sec)).quantize(DP3, rounding=ROUND_HALF_UP)
    leading_fillers = getattr(text_processor, "LEADING_FILLERS", {"嗯", "啊", "呃", "哦", "哎", "喂"})
    omni_segments = []
    full_text = ""
    filtered_count = 0

    for seg in punct_segments:
        logger.info(f"build_omni_response seg={seg}")
        clean_text = text_processor.process(seg["text"], text_lang=src_lang)
        if not clean_text or not clean_text.strip():
            filtered_count += 1
            continue

        start_s_d = Decimal(seg["start_time"]).quantize(DP3)
        end_s_d = Decimal(seg["end_time"]).quantize(DP3)
        abs_start_d = vad_start_d + start_s_d
        abs_end_d = vad_start_d + end_s_d
        duration_d = abs_end_d - abs_start_d

        pure_text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", clean_text)
        pure_len = len(pure_text)

        if pure_text in leading_fillers and duration_d > Decimal("0.8"):
            filtered_count += 1
            continue
        if pure_len <= 2 and duration_d > Decimal("2.0"):
            filtered_count += 1
            continue
        if duration_d < MIN_DURATION:
            filtered_count += 1
            continue


        omni_segments.append({
            "segment_index": segment_index_offset + len(omni_segments),
            "abs_start_time": float(abs_start_d),
            "abs_end_time": float(abs_end_d),
            "start_time": float(start_s_d),
            "end_time": float(end_s_d),
            "text": clean_text,
            "wordsSeg": seg.get("chars", []),
        })

        full_text += clean_text

    return {
        "sentence": full_text,
        "segments": omni_segments,
        "filtered_count": filtered_count,
        "punct_segment_count": len(punct_segments),
        "infer_time": infer_time
    }

