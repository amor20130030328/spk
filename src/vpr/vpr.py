from src.vpr.online_clustering import OnlineClustering
from pyannote.core import Segment, Timeline, SlidingWindow
import re
import numpy as np
from collections import defaultdict
from decimal import Decimal
import time
import os
from src.vpr.feature.fileio import read_hyperyaml
from src.configs.config import config

TS_RE = re.compile(r"\[(\d+\.\d+)\]")
class MeetingTextProcessor:
    """
    会议语音文本后处理核心组件。
    负责：降噪、语种拦截、语气词清洗、口吃修复、标点与排版优化。
    """
    def __init__(self):
        # 基础标点清洗正则
        self.RE_CHINESE_PUNC_REPEAT = re.compile(r"([，。！？；：”“’‘（）【】《》—…])\1{1,}")
        self.RE_PUNC_SURROUND_SPACE = re.compile(r"\s*([，。！？；：”“’‘（）【】])\s*")
        self.RE_LEADING_PUNC = re.compile(r"^[，。！？；：]+")

        # 语种与幻觉拦截正则
        self.ALLOWED_CHAR_PATTERN = re.compile(
            r'^[\u4e00-\u9fffA-Za-z0-9\s'
            r'，。！？；：、“”‘’（）《》【】—…·'
            r',.!?;:\'"()\[\]{}<>/\-+_=@#$%^&*~|\\`]+$'
        )
        self.NOISE_PATTERN = re.compile(
            r"^(啪|咚|哒|砰|嚓|嘟|嗡|嘶|咳)+$|^\[?(咳嗽|cough|click|sil|noise|听不清|静音|掌声|笑声|音乐|杂音)\]?$"
        )

        # 语气词与英文停用词配置
        self.FILLER_WORDS_EN = re.compile(r'\b(um|uh|mhm|ah|uh-?huh|er|uhh|mm|e?h)\b', re.IGNORECASE)
        # self.CONFIRM_WORDS = {"好", "对", "嗯", "嗯嗯", "是的", "没问题", "行", "可以", "好的", "OK", "ok"}
        self.CONFIRM_WORDS = {"好", "对", "嗯嗯", "是的", "没问题", "行", "可以", "好的", "OK", "ok"}
        self.LEADING_FILLERS = {"嗯", "啊", "呃", "哦", "哎", "喂"}

    def _clean_punctuation(self, text: str) -> str:
        """步骤 A: 基础字符与特殊 Tag 清理"""
        text = text.replace("：", "，").replace("\"", "").replace("'", "")
        text = text.replace("《", "").replace("》", "").replace("**", "")
        text = text.replace("<unk>", "").replace("...", "…")
        text = text.replace("<noise>", "")  # 清除 ASR 引擎自带的噪声 tag

        text = self.RE_PUNC_SURROUND_SPACE.sub(r"\1", text)
        text = self.RE_CHINESE_PUNC_REPEAT.sub(r"\1", text)
        text = self.RE_LEADING_PUNC.sub("", text)
        return text

    def _filter_other_languages(self, text: str) -> str:
        """步骤 B: 纯净度防御，拦截包含日、韩、泰等小语种字符的句子"""
        if not self.ALLOWED_CHAR_PATTERN.fullmatch(text):
            return ""
        return text

    def _rule_filler_char(self, text: str) -> str:
        """步骤 D-1: 语气词精准清理策略"""
        pure_text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)

        # 1. 独立成句的确认（Backchanneling），保留真实回应
        if pure_text in self.CONFIRM_WORDS:
            return text

        # 2. 剔除句首无意义的连续缓冲词
        pure_sound_fillers = r"(?:嗯|啊|呃|哦|哎(?!呀|哟|呦|唷))"
        filler_pattern = rf"^({pure_sound_fillers}[，。！？\s]*)+"
        text = re.sub(filler_pattern, "", text).strip()

        # 3. 句中口吃停顿保护与替换
        for f in ["嗯", "啊", "呃", "哦", "哎"]:
            if f == "哎":
                # 保护“哎呀”、“哎呦”等有实际情绪的词汇
                text = re.sub(rf'(?<=[^\s，。！？])[，\s]*哎(?![呀哟呦唷])[，\s]*(?=[^\s，。！？])', '，', text)
            else:
                text = re.sub(rf'(?<=[^\s，。！？])[，\s]*{f}[，\s]*(?=[^\s，。！？])', '，', text)

        # 4. 清理残留句首标点
        return self.RE_LEADING_PUNC.sub("", text.strip())

    def _rule_disfluency(self, text: str) -> str:
        """步骤 D-2: 重复词与口吃精准修复"""
        # 积极情绪白名单：允许保留最多 3 次
        text = re.sub(r"([对好行是可])\1{1,}", lambda m: m.group(1) * min(len(m.group(0)), 3), text)

        # 常见无意义结巴：严打，强制截断为 2 次
        stutter_pattern = r"([我你他她这那就的了啊嗯呃哎哦])\1{1,}"
        text = re.sub(
            stutter_pattern,
            lambda m: m.group(1) * 2,
            text
        )

        # 多字连读结巴收敛
        for stutter in ["那个", "就是", "然后", "其实", "是不是"]:
            text = re.sub(rf"({stutter}){{2,}}", stutter, text)
        return text

    def _apply_typography(self, text: str) -> str:
        """步骤 D-3: 中英数平滑排版"""
        if not text: return text
        # 修复大写缩写带空格问题 (O S -> OS)
        pattern = r'(?<![a-zA-Z])([A-Z](?:\s+[A-Z])+)(?![a-zA-Z])'
        text = re.sub(pattern, lambda m: m.group(1).replace(' ', ''), text)
        # 中英文/数字交界处增加半角空格
        text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z0-9])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fa5])', r'\1 \2', text)
        return text

    def _filter_short_fragments(self, text: str) -> str:
        """步骤 E-1: 极短碎片抛弃，精准狙击单字幻觉"""
        pure_text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)

        if len(pure_text) >= 2:
            return text

        if len(pure_text) == 1:
            # 严格单字白名单
            # valid_single_chars = {"好", "对", "嗯", "行", "是", "能", "有", "懂"}
            valid_single_chars = {"好", "对", "行", "是", "能", "有", "懂"}
            if pure_text not in valid_single_chars:
                return ""

        if not pure_text:
            return ""

        return text

    def process(self, text: str, text_lang: str = "zh") -> str:
        """终极主处理 Pipeline (防漏、防幻觉、防乱码)"""
        if not text: return ""

        # A. 基础清洗 (移除 <noise>, <unk> 等，避免干扰后续正则)
        text = self._clean_punctuation(text)

        # B. 语种拦截 (纯净度防御)
        text = self._filter_other_languages(text)
        if not text: return ""

        # C. 幻觉精准狙击 (防纯噪音拟声词)
        text_for_noise_check = re.sub(r"[，。！？；：\s]+", "", text)
        if self.NOISE_PATTERN.match(text_for_noise_check):
            return ""

        # D. 核心化学清洗
        if text_lang == "zh":
            # text = self._rule_filler_char(text)
            # text = self._rule_disfluency(text)
            text = self._apply_typography(text)
        else:
            text = re.sub(r"\([^)]*\)", "", text)
            text = re.sub(r"\[[^\]]*\]", "", text)
            text = self.FILLER_WORDS_EN.sub('', text)

        # E. 收尾与极短片段过滤
        text = re.sub(r"\s+", " ", text).strip()
        text = self.RE_LEADING_PUNC.sub("", text)
        text = self._filter_short_fragments(text)

        # F. 最终下限防御 (确保输出内容至少包含一个有意义的中/英/数字)
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
            return ""

        return text

TEXT_PROCESSOR = MeetingTextProcessor()

def truncate_repeated_text(text, max_keep=2, repeat_threshold=5):
    """
    按标点分隔文本块，找到连续重复达5次的文本块，截断到其第二次出现完成的位置
    :param text: 原始文本
    :param max_keep: 保留的重复次数（固定为2）
    :param repeat_threshold: 触发截断的重复次数阈值（固定为5）
    :return: 截断后的文本、第二次出现完成的字符位置
    """
    if not text:
        return text

    # 1. 拆分：文本块+标点/空格（保留分隔符，用于还原字符位置）
    separators = r'([，。！？；\s、：…—])'
    parts = re.split(separators, text)
    parts = [p for p in parts if p]  # 过滤空字符串

    if len(parts) < 2:
        return text

    # 2. 遍历：统计连续重复，记录第二次完成的字符位置
    char_pos = 0          # 实时追踪原始文本的字符位置
    truncate_pos = None   # 第二次出现完成的字符位置
    prev_text = None      # 上一个非标点文本块
    repeat_count = 0      # 连续重复次数
    trigger_truncate = False  # 是否触发截断（重复达5次）

    for part in parts:
        # 累加当前块的字符长度，更新原始文本位置
        char_pos += len(part)

        # 只处理非标点/空格的文本块
        if not re.match(separators.strip('()'), part):
            if part == prev_text:
                # print(part)
                repeat_count += 1
                # 重复达阈值，标记需要截断
                if repeat_count >= repeat_threshold:
                    trigger_truncate = True
                # 记录第二次出现完成的位置（仅第一次触发时记录）
                if repeat_count == max_keep:
                    truncate_pos = char_pos  # 此时char_pos是第二次完成的位置
            else:
                # 新文本块，重置计数
                prev_text = part
                repeat_count = 1
                truncate_pos = None
                trigger_truncate = False  # 重置截断标记

        # 找到截断位置后，立即终止遍历
        if trigger_truncate and truncate_pos is not None:
            break

    # 3. 执行截断：有截断位置则截断，否则返回原文本
    if truncate_pos is not None and truncate_pos > 0:
        truncated_text = text[:truncate_pos].strip()
    else:
        truncated_text = text
        truncate_pos = len(text)

    return truncated_text

class Vpr:
    def __init__(self, vpr_actor, itn_actor, logger):
        self.dia_threshold = 0.36 #0.58
        self.cls_update_len = 0.36 #0.58
        self.online_clustering = OnlineClustering(self.dia_threshold, self.cls_update_len)
        self.global_max_weight = 0
        self.sess = vpr_actor.vpr_base_model
        self.label_mapping = {}  # 记录内部聚类ID到连续ID的映射关系
        self.current_speaker_id = 1  # 连续发号器（想从1开始就设为1）
        self.global_max_weight = 0
        self.last_valid_label = None
        self.prev_segment_embedding = None
        self.prev_segment_end_time  = 0.0
        self.prev_segment_quality   = 0.0
        self.prev_segment_duration  = 0.0      # 上一段时长（秒）
        self.prev_cluster_label     = None     # 上一段最终归属的簇标签
        self.max_time_gap           = 5.0
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.itn_actor = itn_actor
        self.yaml_path = os.path.join(script_dir, "feature", "train.yaml")
        print("self.yaml_path ", self.yaml_path)

        # ONNX 模型吃的是 input_features，所以这里仍然需要原始 feature_extractor
        self.feature_extractor = vpr_actor.feature_extractor
        self.logger = logger

        self.sr = int(vpr_actor.sample_rate)
        self.max_len = int(vpr_actor.dur_range[-1] * vpr_actor.sample_rate)


    def _wav_to_input_features(self, data):
        data = data[:self.max_len]
        st = time.time()
        features = self.feature_extractor(
            data,
            sampling_rate=self.sr,
            return_tensors="np",
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        input_features = features.input_features.astype(np.float32)
        self.logger.info(f"input_features cost_time: {time.time() - st}")
        return input_features
    
    # def _wav_to_input_features(self, data):
    #     """
    #     wav segment -> input_features（向量化优化版）。
    #     等价于 SeamlessM4TFeatureExtractor.__call__，但用批量 FFT 替代逐帧循环。
    #     """
    #     data = data[:self.max_len]
    #
    #     # ---- 参数（与 SeamlessM4TFeatureExtractor 完全一致）----
    #     frame_length = 400
    #     hop_length = 160
    #     fft_length = 512
    #     window = self.feature_extractor.window.astype(np.float64)  # povey window (400,)
    #     mel_filters = self.feature_extractor.mel_filters  # (257, 80)
    #     mel_floor = 1.192092955078125e-07
    #     preemphasis = 0.97
    #     stride = 2
    #
    #     # ---- Kaldi compliance: float32 → int16 scale ----
    #     waveform = np.squeeze(data) * (2 ** 15)
    #     waveform = waveform.astype(np.float64)
    #
    #     # ---- 向量化分帧（center=False，不做 padding）----
    #     num_frames = 1 + (len(waveform) - frame_length) // hop_length
    #     if num_frames <= 0:
    #         waveform = np.pad(waveform, (0, frame_length - len(waveform)))
    #         num_frames = 1
    #
    #     # 用索引矩阵一次性取出所有帧 → (num_frames, frame_length)
    #     frame_indices = np.arange(num_frames)[:, None] * hop_length + np.arange(frame_length)[None, :]
    #     frames = waveform[frame_indices].copy()  # copy 因为后续要原地修改
    #
    #     # ---- 向量化逐帧处理（原来在 for 循环内逐帧做）----
    #     # 1. remove_dc_offset
    #     frames -= frames.mean(axis=1, keepdims=True)
    #     # 2. preemphasis
    #     frames[:, 1:] -= preemphasis * frames[:, :-1]
    #     frames[:, 0] *= (1 - preemphasis)
    #     # 3. windowing
    #     frames *= window
    #
    #     # ---- 批量 FFT（核心加速点：一次 rfft 替代 198 次单独 rfft）----
    #     fft_buffer = np.zeros((num_frames, fft_length), dtype=np.float64)
    #     fft_buffer[:, :frame_length] = frames
    #     spectrogram = np.fft.rfft(fft_buffer, axis=1)  # (num_frames, 257) complex
    #
    #     # power = 2.0（直接用 real²+imag² 避免 abs 再平方的 sqrt 开销）
    #     spectrogram = (spectrogram.real ** 2 + spectrogram.imag ** 2).astype(np.float64)
    #
    #     # mel filters: (num_frames, 257) @ (257, 80) → (num_frames, 80)
    #     spectrogram = np.maximum(mel_floor, spectrogram @ mel_filters)
    #     spectrogram = np.log(spectrogram)  # log_mel="log"
    #
    #     # ---- CMVN normalize per mel bins ----
    #     spectrogram = (spectrogram - spectrogram.mean(axis=0, keepdims=True)) / \
    #                   np.sqrt(spectrogram.var(axis=0, ddof=1, keepdims=True) + 1e-7)
    #
    #     # ---- stride reshape: (1, num_frames, 80) → (1, num_frames//2, 160) ----
    #     remainder = num_frames % stride
    #     if remainder != 0:
    #         spectrogram = spectrogram[:num_frames - remainder, :]
    #         num_frames = num_frames - remainder
    #
    #     input_features = spectrogram[np.newaxis, :, :]  # (1, num_frames, 80)
    #     input_features = np.reshape(input_features, (1, num_frames // stride, 80 * stride))
    #     input_features = input_features.astype(np.float32)
    #
    #     return input_features

    def _run_onnx(self, input_features):
        """
        input_features -> embedding。
        """
        st = time.time()
        emb = self.sess.forward(np.ascontiguousarray(input_features))
        emb = np.squeeze(emb).astype(np.float32)
        et = time.time()
        print("_run_onnx",et-st)

        return emb

    def make_embedding(self, wav_path, segline):
        """
        从 wav 中截取 segment。
        """
        # full_waveform = self._load_full_wav(wav_path)
        full_waveform = wav_path

        cur_start_frame = round(segline[0].start * self.sr)
        cur_end_frame = round(segline[-1].end * self.sr)
        cur_waveform = full_waveform[cur_start_frame: cur_end_frame]
        if cur_waveform.shape[-1] < 160:
            return [], [], []

        # data = np.asarray(data, dtype=np.float32)

        batch_data = []
        weights = []
        seg_segs = []
        for i, subseg in enumerate(segline):
            # 与 VAD 的交集部分
            start_frame = round(subseg.start * self.sr)
            num_frames = round(subseg.duration * self.sr)
            data = full_waveform[start_frame : start_frame + num_frames]

            if data.shape[-1] < 160:   # 有效音频阈值（例如 10ms）
                continue

            # 能量 / 权重计算（向量化）
            energy_i = np.clip(np.mean(data ** 2), a_min=1e-8, a_max=None)
            weight_i = float(np.sqrt(energy_i))   # 转为 Python 标量
            weights.append(weight_i)
            seg_segs.append(subseg)

            curr_len = data.size
            if curr_len < self.max_len:
                repeats = (self.max_len // curr_len) + 1
                # 反射填充：取 data[1:-1] 翻转后拼接到 data 尾部
                reflect_part = np.flip(data[1:-1], axis=0)
                reflect_block = np.concatenate([data, reflect_part], axis=0)
                data = np.tile(reflect_block, repeats)[:self.max_len]
            else:
                data = data[:self.max_len]

            # 模型推理（假设 self.model 接受 numpy 数组并返回 numpy 数组）
            # emb = self.model(data)            # 输出形状可能是 (dim,) 或 (1, dim)
            input_features = self._wav_to_input_features(data)
            emb = self._run_onnx(input_features)
            batch_data.append(np.squeeze(emb).astype(np.float32))

        if len(batch_data) == 0:
            print("!!!!!!!!!!!!!!!!!!!")
            # 没有任何有效子段，合并整个 segline 区间
            start_frame = round(segline[0].start * self.sr)
            end_frame = round(segline[-1].end * self.sr)
            data = full_waveform[start_frame : end_frame]

            if data.shape[-1] < 160:
                return [], [], []

            # emb = self.model(data)
            input_features = self._wav_to_input_features(data)
            emb = self._run_onnx(input_features)
            batch_data.append(np.squeeze(emb).astype(np.float32))
            weights = [1e-8]
            seg_segs.append(Segment(start=segline[0].start, end=segline[-1].end))

        # 归一化嵌入向量
        batch_data = np.array(batch_data)          # shape: (N, D)
        norms = np.linalg.norm(batch_data, axis=1, keepdims=True) + 1e-6
        batch_data /= norms

        return batch_data, weights, seg_segs

    def handle(self, segments, audio_data, vad_start):
        result = []
        last_text = None
        repeat_count = 0
        MIN_DURATION = 0.35
        for idx, segment in enumerate(segments):
            text, start_s, end_s =segment['text'], float(segment['start_time']), float(segment['end_time'])
            clean_text = self.itn_actor.normalize(text)

            # 被拦截或洗空的内容，直接跳过
            if not clean_text  or not clean_text.strip():
                continue

            if clean_text == last_text:
                repeat_count += 1
            else:
                last_text = clean_text
                repeat_count = 1

            if repeat_count > 2:
                continue # 超过2次重复的丢弃

            if end_s <= start_s:
                estimated_duration = max(0.5, len(clean_text) * 0.25)
                end_s = start_s + estimated_duration
                # continue
            if end_s - start_s < MIN_DURATION:
                continue

            duration = end_s - start_s
            self.logger.error(f"vad segment : {start_s} - {end_s} |{duration}: {text}")
            pure_text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", clean_text)
            pure_len = len(pure_text)
            if pure_text in TEXT_PROCESSOR.LEADING_FILLERS and duration > 0.8:
                print(f"[拦截] 发现异常超长单字 (大概率幻觉/呼吸): '{clean_text}', 时长: {duration:.2f}s")
                continue

            # 2. 超长短句幻觉拦截（可选进阶）
            # 如果只识别出 2-3 个字，但时间却超过了 2 秒，也极度可疑
            if pure_len <= 2 and duration > 2.0:
                print(f"[拦截] 发现异常低信息密度短句: '{clean_text}', 时长: {duration:.2f}s")
                continue

            if audio_data is not None:
                raw_label = self.diarization_online(audio_data, start_s, end_s, vad_start)
            else:
                raw_label = -1

            # ================= 新增：连续化映射逻辑 =================
            if raw_label != -1 and raw_label is not None:
                # 如果内部聚类产生了一个新的簇 ID，给它分配下一个连续号码
                if raw_label not in self.label_mapping:
                    self.label_mapping[raw_label] = self.current_speaker_id
                    self.current_speaker_id += 1

                # 获取映射后的连续标签
                mapped_label = self.label_mapping[raw_label]
            else:
                mapped_label = -1

            # --------------------------
            # 保存段信息
            # --------------------------
            result.append({
                "isSpeaker": "true",
                "speaker": mapped_label,
                "word": clean_text,
                "vadInfo": {
                    "start_of_speech": round((vad_start + start_s) * config.SEC_FRAME_SIZE),
                    "end_of_speech": round((vad_start + end_s) * config.SEC_FRAME_SIZE),
                }

            })

        return result


    def split_segment(self, segment, step, duration):
        timeline = Timeline()
        if segment.duration < (duration + step):
            return timeline.add(segment)
        window = SlidingWindow(duration=duration,
                            step=step,
                            start=segment.start,
                            end=segment.end)

        for s in window:
            # ugly hack to account for floating point imprecision
            if s in segment:
                timeline.add(s)
                last = s

        if last.end < segment.end:
            timeline.add(Segment(start=segment.end-duration,
                                end=segment.end))
        return timeline

    def diarization_online(self, audio, start_s, end_s, vad_start):
        seg_start    = float(Decimal(str(vad_start)) + Decimal(str(start_s)))          # 秒
        seg_end      = float(Decimal(str(vad_start)) + Decimal(str(end_s)))            # 秒
        seg_duration = float(Decimal(str(end_s)) - Decimal(str(start_s)))       # 秒
        # 特征提取
        timelines = self.split_segment(Segment(start_s, end_s), 1, 2)
        if len(timelines) == 0:
            fallback_label = self.last_valid_label if self.last_valid_label is not None else 0
            return fallback_label
        confidence = []

        start_time = time.time()
        # print("timelines", timelines)
        seg_embedding, weights, seg_segs = self.make_embedding(audio, timelines)
        if len(weights) == 0:
            #print("no speaker embeddings", start_s, end_s, vad_start)
            # 如果之前有合法的说话人，就用之前的；如果第一句就被过滤了，给一个兜底的 'UNKNOWN'
            fallback_label = self.last_valid_label if self.last_valid_label is not None else 0
            # 注意：因为 weights 为 0，seg_segs 也是空的，这里直接使用外层的 segment 时间边界
            return fallback_label
        current_max = max(weights)
        if current_max > self.global_max_weight:
            self.global_max_weight = current_max
        else:
            # 缓慢衰减（比如每收到一个新片段衰减 1%），这样如果大声的人走了，参考值也会慢慢降下来适应远场
            self.global_max_weight = self.global_max_weight * 0.99
        end_time = time.time()
        #print(f"耗时 : {end_time - start_time} {vad_start}" )
        # self.global_max_weight = max(self.global_max_weight, max(weights))
        confidence = []
        n_sub           = len(seg_embedding)
        # if len(seg_embedding) > 1:
        if n_sub > 1:
            weights_np = np.array(weights).reshape(-1, 1)
            total_weight = np.sum(weights_np) + 1e-6
            # weighted_sum: 所有向量指向方向的叠加
            weighted_sum = np.sum(np.array(seg_embedding) * weights_np, axis=0)
            mean_vector = weighted_sum / total_weight
            consistency = np.linalg.norm(mean_vector)
            cluster_center = mean_vector / (consistency + 1e-6)
            purity_factor = 1.0
            if consistency < 0.45:
                purity_factor = max(0.0, (consistency - 0.1) / 0.25) ** 2

            micro_sim = np.dot(seg_embedding, cluster_center)
            SIM_MIN = 0.10  # 边缘质量分数
            SIM_MAX = 0.50  # 极高质量分数

            # 将 micro_sim 线性映射到 [0.1, 1.0] 的区间
            mapped_sim = 0.1 + 0.9 * (micro_sim - SIM_MIN) / (SIM_MAX - SIM_MIN)
            # 限制范围，防止爆炸或负数
            mapped_sim = np.clip(mapped_sim, 0.1, 1.0)
            # final_conf = purity_factor * micro_sim
            final_conf = purity_factor * mapped_sim
            final_conf = np.clip(final_conf, 0.0, 1.0)
            confidence = final_conf.tolist()
        elif n_sub == 1:
            t_score = min(seg_segs[0].duration / 2.0, 1.0)
            e_score = min(weights[0] / (self.global_max_weight + 1e-6), 1.0)
            combined_quality = 0.7 * t_score + 0.3 * e_score
            conf_val = combined_quality * 0.45 + 0.15
            # conf_val = t_score
            confidence = [conf_val]
        else:
            fallback_label = self.last_valid_label if self.last_valid_label is not None else 0
            return fallback_label

        temp_label = []

        for x, seg, conf in zip(seg_embedding, seg_segs, confidence):
            data = {
                'embedding': x,
                'duration':  seg_duration*100,
                'start': seg_start*100,
                'end': seg_end*100,
                'confidence': float(conf),
                # 'weight': float(w),
            }
            # 实时聚类
            label = self.online_clustering.update(data)
            # print(data, label)
            # if label is None:
            #     continue

            if conf > 0.2  and label != None:
                temp_label.append((label, data['duration']))

        if not temp_label:
            fallback_label = self.last_valid_label if self.last_valid_label is not None else 0
            return fallback_label

        duration_map = defaultdict(int)
        for label, duration in temp_label:
            if label is not None:  # 排除None标签
                duration_map[label] += duration

        if not duration_map:
            # return -1
            fallback_label = self.last_valid_label if self.last_valid_label is not None else 0
            return fallback_label

        label = max(duration_map, key=duration_map.get)
        # print(label)
        self.last_valid_label = label
        return label



