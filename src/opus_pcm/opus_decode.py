import ctypes
import os
import struct

import opuslib
from src.logger.logger_adapter import logger


class OpusToPcmConverter:
    def __init__(self, sample_rate=16000, channels=1, session_id='opus'):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.channels = channels
        # 每 20ms 的采样数 (对于 16kHz 是 320)
        self.samples_20ms = int(sample_rate * 0.02)
        # 每 40ms 的采样数 (对于 16kHz 是 640)
        self.samples_40ms = int(sample_rate * 0.04)
        self._load_opus_lib()

        # 初始化解码器
        try:
            self.decoder = opuslib.Decoder(self.sample_rate, self.channels)
        except Exception as e:
            logger.error(f'[{self.session_id}] init CustomOpusDecoder error {e}', exc_info=True)
            raise RuntimeError(f"init CustomOpusDecoder error: {str(e)}") from e

    def decode_short_opus(self, frame_data: bytes) -> bytes:
        """
        解析 4 + (2+V1) + (2+V2) 格式
        """
        if len(frame_data) < 8:  # 最小长度校验: 4(head) + 2(len1) + 2(len2)
            return b""

        try:
            # 解析 V1
            v1_len = struct.unpack_from('!H', frame_data, 4)[0]
            v1_start = 6
            v1_end = v1_start + v1_len
            v1_data = frame_data[v1_start:v1_end]

            # 解析 V2
            if len(frame_data) < v1_end + 2:
                return b""

            v2_len = struct.unpack_from('!H', frame_data, v1_end)[0]
            v2_start = v1_end + 2
            v2_end = v2_start + v2_len
            v2_data = frame_data[v2_start:v2_end]

            # 解码并合并 (使用 list 提高拼接效率)
            chunks = []
            for data in [v1_data, v2_data]:
                if data:
                    chunks.append(self.decoder.decode(data, self.samples_20ms))
                else:
                    # 补静音：1个采样2字节(16bit PCM) * 采样数
                    chunks.append(b'\x00' * (self.samples_20ms * self.channels * 2))

            return b"".join(chunks)

        except (struct.error, opuslib.OpusError) as e:
            logger.error(f'[{self.session_id}] decode_short_opus error {e}', exc_info=True)
            return b'\x00' * (self.samples_40ms * self.channels * 2)

    def decode_long_opus(self, frame_data: bytes) -> bytes:
        """
        解析 2字节长度 + V (40ms) 格式
        """
        if len(frame_data) < 2:
            return b""

        try:
            v_len = struct.unpack('!H', frame_data[0:2])[0]
            v_data = frame_data[2: 2 + v_len]
            return self.decoder.decode(v_data, self.samples_40ms)
        except Exception as e:
            logger.error(f'[{self.session_id}] decode_long_opus error {e}', exc_info=True)
            return b'\x00' * (self.samples_40ms * self.channels * 2)

    def _load_opus_lib(self):
        paths_to_check = [
            '/usr/lib64/libopus.so',
            '/usr/local/lib/libopus.so',
            'libopus.so.0'
        ]

        for path in paths_to_check:
            if path and (os.path.exists(path) or '/' not in path):
                try:
                    ctypes.CDLL(path)
                    return True
                except Exception as e:
                    logger.error(f"load_opus_lib error: {path}, {str(e)}")
        return False
