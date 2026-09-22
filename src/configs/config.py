# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import json
import os
from typing import Final
from src.logger.logger_adapter import logger
from src.vad.silero import SileroVadOptions

cur_path = os.path.dirname(__file__)
sfs_model_base_dir = ""
if os.getenv('MODEL_SFS') and os.getenv('MODEL_OBJECT_ID'):
    sfs_info = json.loads(os.getenv('MODEL_SFS'))
    sfs_model_base_dir = os.path.join(sfs_info['sfsBasePath'], os.getenv('MODEL_OBJECT_ID'))
model_relative_dir = str(os.getenv('MODEL_RELATIVE_DIR', 'model'))
model_dir = os.path.join(sfs_model_base_dir, model_relative_dir)
logger.debug(f'model_dir is {model_dir}')



def get_vad_options():
    return SileroVadOptions(
        # Threshold for what is considered speech (default 0.5)
        threshold=0.5,
        # Final speech chunks shorter min_speech_duration_ms are thrown out (default 250)
        min_speech_duration_ms=250,
        # Max duration of speech chunks, longer will be split (default float('inf'))
        max_speech_duration_s=30,
        # Wait for ms at the end of each speech chunk before separating it (default 2000)
        min_silence_duration_ms=50,
        # Chunk size for VAD model. Can be 512, 1024, 1536 for 16k s.r. (default 1024)
        window_size_samples=1024,
        # Final speech chunks are padded by speech_pad_ms each side (default 400)
        speech_pad_ms=400,
    )


def str_to_bool(s: str) -> bool:
    """将字符串转换为布尔值，处理常见的真/假标识"""
    s_lower = s.lower()
    if s_lower in ("true", "1"):
        return True
    elif s_lower in ("false", "0"):
        return False
    else:
        return True


class Config:
    """全局配置类，用于集中管理所有可调参数"""

    max_connections = 16

    ACTOR_RESET_TIMEOUT_S: Final[float] = 3.0

    # System Behavior
    RECEIVE_QUEUE_MAX_SIZE: Final[int] = 250
    ASYNC_QUEUE_WAIT_TIMES: Final[float] = 20.0
    MONITOR_SLEEP_TIME: Final[float] = 0.5

    WS_TUNNEL_ID_FIELD: Final[str] = "tunnelId"
    WS_CONNECTION_FIELD: Final[str] = "connected"

    SESSION_TIMEOUT = 30  # 会话5秒无消息收发则认为超时

    vad_options = get_vad_options()

    ENERGY_THRESHOLD = 0.0001

    model_dir = model_dir
    vad_path = os.path.join(model_dir, "vad.onnx")
    vpr_path = os.path.join(model_dir, "w2vbert_spk_model_opt_graph.mindir")
    itn_path = os.path.join(model_dir, "itn/zh/itn")

    omni_address = os.environ.get("omni_address", "")
    trans_address = os.environ.get("trans_address", "")
    mep_app_id = os.environ.get("mep_app_id", "")
    mep_sign_key = os.environ.get("mep_sign_key", "")
    speaker_omni_bid = os.environ.get("speaker_omni_bid", 'qwen3_omni_30b_speakerlog')
    speaker_omni_flowId = os.environ.get("speaker_omni_flowId", 'qwen3_omni_30b_speakerlog')
    qwen3_asr_bid = os.environ.get("qwen3_asr_bid", 'speaker_engine_asr_test')
    qwen3_asr_flowId = os.environ.get("qwen3_asr_flowId", 'speaker_engine_asr_test')

    sample_rate = 16000
    # 16-bit PCM最大值，用于音频归一化 (int16 -> float32)
    audio_normalization_factor = 32768.0

    is_debug = os.environ.get("is_debug", True)
    omni_model = os.environ.get("omni_model", "qwen3_omni")
    asr_model = os.environ.get("asr_model", "qwen3_asr")
    fa_model = os.environ.get("fa_model", "qwen3_fa")
    debug_device_Id = os.environ.get("debug_device_Id", "amore")

    vpr_dia_threshold = os.environ.get("vpr_dia_threshold", 0.58)  #
    vpr_cls_update_len = os.environ.get("vpr_cls_update_len", 0.58)  #





    FRAME_DURATION_MS = 32  # 帧时长(ms)
    FRAME_SIZE = int(sample_rate * FRAME_DURATION_MS / 1000)  # 每帧样本数
    
    # VAD参数
    max_segment_duration = 18.0  # 最大段时长(s)
    MIN_SEGMENT_DURATION = 0.4  # 最小段时长(s)
    EXPECTED_SIZE = 512

    MIN_TARGET_DURATION = 10.0  # 最小目标时长(s)：达到此长度后，遇到静音即可切分
    MAX_TARGET_DURATION = 14.0  # 最大强制时长(s)：达到此长度哪怕在说话也强制切分

    MS_SAMPLE = sample_rate / 1000
    SEC_FRAME_SIZE=100
    MAX_DECODE_TIME = 25
    WAIT_TIME = 0.04

    # ===== Guard 幻觉防护配置 =====
    GUARD_REPEAT_THRESHOLD = 6           # 重复模式检测阈值（连续重复字符数）
    GUARD_MAX_CHARS_PER_SEC = 12.0       # 最大字符速率（字/秒）
    GUARD_MAX_DELTA_CHARS = 20           # 与可信文本最大差异字符数
    GUARD_SINGLETON_RATIO = 0.7          # 单字符占比阈值
    GUARD_SINGLETON_MIN_LEN = 6          # 触发单字符检测的最小文本长度
    GUARD_HALLUCINATION_STREAK_RESET = 2  # 连续拒绝多少次后重置可信文本




# 创建一个全局配置实例
config = Config()



