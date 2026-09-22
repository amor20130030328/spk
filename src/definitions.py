# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import time
from dataclasses import dataclass
from typing import NamedTuple
from typing import Optional, Union, Dict
import numpy as np
from pydantic import BaseModel


@dataclass
class ExtInfo:
    activate_session: Optional[int] = 0

@dataclass
class SpeakerLog:
    frame: Optional[np.ndarray] = None  # np.int16 音频数据
    text: Optional[str] = ""
    start: Optional[int] = None
    end: Optional[int] = None



@dataclass
class AsrResult:
    text: Optional[str] = ""
    isFinish: Optional[bool] = False
    retType: Optional[str] = "partial"


@dataclass
class AsrResult:
    text: Optional[str] = ""
    isFinish: Optional[bool] = False
    retType: Optional[str] = "partial"


@dataclass
class AsrSpeakerResult:
    g_bg: Optional[str] = ""
    g_ed: Optional[str] = ""
    infer_time: Optional[str] = ""



@dataclass
class TimeInfo:
    g_bg: Optional[str] = ""
    g_ed: Optional[str] = ""
    infer_time: Optional[str] = ""



@dataclass
class SpeakerLongResponse:
    action: Optional[str] = "partial"
    errorCode: Optional[int] = 0
    errorMsg: Optional[str] = "Success"
    sessionId: Optional[str] = ""
    deviceId: Optional[str] = ""
    vendor: Optional[str] = None
    omni: Optional[Dict] = None
    result: Optional[Union[AsrResult, AsrSpeakerResult]] = None
    ext_info: Optional[ExtInfo] = None
    time_info : Optional[TimeInfo] = None



@dataclass
class VadResult:
    start: Optional[int] = None
    end: Optional[int] = None
    isFinal: Optional[bool] = None
    isEnd: Optional[bool] = None


class AudioFormat(BaseModel):
    compress: str = "pcm"  # 支持"pcm"/"opus"
    sampleRate: int = 16000
    format: str = "pcm"  # 没啥用
    channel: int = 1
    bitRate: int = 16
    packageCycle: int = 16



class InitialSpeakerRequest(BaseModel):
    sessionId: str
    deviceId: str
    messageName: str
    deviceCategory: str
    audioType: str
    language: str
    allowSaveData: bool
    packageName: str
    audioFormat: AudioFormat




class SParams(NamedTuple):
    chunk_size: float
    duration_thd: float
    compression_thd: float
    logprob_thd: float
    silence_thd: float
    prevent_duration: float
    prevent_text_len: dict
    partial_ratio: dict
    max_duration: float
    try_punc_thd: float
    deley_thd: float


class MonitorMetrics(BaseModel):
    messagesReceived: int = 0
    audioReceived: int = 0
    messagesSent: int = 0
    errors: int = 0
    startTime: float = time.time()
    lastActivity: float = time.time()

    def __str__(self):
        return (
            f"MonitorMetrics(messagesReceived={self.messagesReceived}, "
            f"audioReceived={self.audioReceived}, "
            f"messagesSent={self.messagesSent}, "
            f"errors={self.errors}, "
            f"startTime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.startTime))}, "
            f"lastActivity={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.lastActivity))})"
        )



