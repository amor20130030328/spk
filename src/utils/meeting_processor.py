import re

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
            r',.!?;:\'"()\[\]{}<>/\-+_= @#$%^&*~|`]+$'
        )
        self.NOISE_PATTERN = re.compile(
            r"^(啪|咚|哒|砰|嚓|嘟|嗡|嘶|咳)+$|^\[?(咳嗽|cough|click|sil|听不清|静音|掌声|笑声|音乐|杂音)\]?$"
        )
        
        # 语气词与英文停用词配置
        self.FILLER_WORDS_EN = re.compile(r'\b(um|uh|  mhm|ah|uh-?huh|er|uhh|mm|e?h)\b', re.IGNORECASE)
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
            text = self._rule_filler_char(text)
            text = self._rule_disfluency(text)
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
processor = MeetingTextProcessor()
