import os.path

from src.configs.config import config

class MonitorManager:

    MODEL_PATH = config.model_dir
    AUDIO_PATH = os.path.join(MODEL_PATH, "audio")

    def __init__(self, session_id, device_id, logger):
        self.session_id = session_id
        self.device_id = device_id
        self.logger = logger
        self.logger.info(f"MonitorManager init sessionId = {self.session_id}, deviceId = {self.device_id}")
        self.frame_count = 0
        self.save_path = os.path.join(self.AUDIO_PATH, self.session_id)
        self._ensure_save_dir()

    def _ensure_save_dir(self):
        """确保保存目录存在"""
        if not os.path.exists(self.AUDIO_PATH):
            os.makedirs(self.AUDIO_PATH, exist_ok=True)

    def _is_online(self):
        """确保保存目录存在"""
        return not config.is_debug

    def save_bytes_to_file(self, data: bytes, audio_type: str, mode: str = 'ab') -> bool:
        """保存字节数据到文件
        Args:
            data: 要保存的字节数据
            audio_type: 音频类型（opus/pcm等）
            mode: 打开模式，默认 'ab'（二进制追加）
        Returns:
            bool: 保存是否成功
        """
        if self._is_online():
            return

        try:
            self.frame_count += 1
            file_path = f"{self.save_path}.{audio_type}"
            with open(file_path, mode) as f:
                f.write(data)
            self.logger.info(f"数据已保存到 {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存文件失败 {file_path}: {e}")
            return False






