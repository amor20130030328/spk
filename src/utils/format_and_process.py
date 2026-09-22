import re
import string
import time

from src.configs.common_punctuation import chinese_punctuation, simple_punctuation, ZH2EN_PUNC_MAP, PUNCS
from src.logger.logger_adapter import logger
from src.utils.filler_and_disfluency import _rule_disfluency, _rule_filler_char, _clean_en_filler_words


def _remove_zh(text: str) -> str:
    """
    去除字符串中的中文字符和中文标点符号。
    """
    # 匹配中文字符、中文标点和全角字符
    pattern = re.compile(r'[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]')
    no_zh_text = pattern.sub(' ', text)
    no_zh_text = re.sub(r'\s+', ' ', no_zh_text)
    return no_zh_text.rstrip()


PUNC_PRIORITY_MAP = {p: i for i, p in enumerate(PUNCS["zh"])}

# 匹配连续中文标点
REMOVE_PUNC_PATTERN = re.compile(r"[？！。，、]{2,}")

# 因为中英混合场景的口水词处理导致的英文标点与中文标点混合
REMOVE_AUTO_PUNC_PATTERN = re.compile(r"[?!.,]\s([？！。，、])")


def _remove_repeat_punc(text: str) -> str:
    def replace(match):
        chars = match.group()
        # 按优先级选出最优字符
        best = min(chars, key=lambda c: PUNC_PRIORITY_MAP.get(c, float('inf')))
        return best

    def replace2(match: re.Match) -> str:
        chinese_punc = match.group(1)
        english_punc = ZH2EN_PUNC_MAP.get(chinese_punc, ",")

        return english_punc + " "

    text = REMOVE_PUNC_PATTERN.sub(replace, text)
    text = REMOVE_AUTO_PUNC_PATTERN.sub(replace2, text)
    return text


chinese_punct = "，。、？；：！"
chinese_punct_with_space_pattern = re.compile(
    f"([{re.escape(chinese_punct)}])\\s+")

SIMPLE_INVERT_CHAR = ["昇"]


def _format_en_text(text, prefix_last_char: str | None = ""):
    # 英文结果不保留中文字符
    text = _remove_zh(text)
    # 英文结果有概率出现Markdown语法的加粗符号，case: We **find out** a good way
    text = text.replace("**", "")
    # case: (Music) I'll start to
    text = _remove_brackets_and_content(text)
    # 去除英文口水词, case: Um, I don't think, mhm, it's a problem
    text = _clean_en_filler_words(text, prefix_last_char)
    return text


def _format_zh_text(text: str, prefix_last_char: str | None = ""):
    is_head_punc = text and text[0] in chinese_punctuation
    # 去除中文的speech quote， e.g. 啊！车队跟汉密尔顿说：“啊，在第十一位啊。”
    text = text.replace("：", "，").replace(
        "“", "").replace("”", "").replace("《", "").replace("》", "").replace("**", "")

    # 去除口水词、乱解码词
    text = _rule_filler_char(text)
    text = _rule_disfluency(text)

    # 去除异常的空格 e.g. ，亡国兴废在此一役。 哎哎，卡住了的怎么回事？原来啊。
    text = chinese_punct_with_space_pattern.sub(r"\1", text)

    # 去除连续的重复标点
    text = _remove_repeat_punc(text)
    # 开源三方件原因，删除简体处理

    if is_head_punc:
        return text
    # 如果句首原本没有标点，而处理后有了标点，则可能是去除口水词残留的
    # bad case: '了。' -> '。'
    if text and len(text) > 1 and text[0] in chinese_punctuation:
        # 如果prefix句尾没有标点，则不去掉句首标点
        if prefix_last_char and prefix_last_char[0] not in simple_punctuation:
            return text
        # 场景1：前一句话句尾有标点
        # 场景2：丢失前一句的信息
        text = text[1:]
    return text


def _remove_brackets_and_content(text):
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    return text


def _format_zh_or_keep_other_langs(text, prefix_last_char):
    remove_zh_str = _remove_zh(text)
    if len(text) != len(remove_zh_str):
        # 有中文字符，则走中文处理
        return _format_zh_text(text, prefix_last_char)
    text = text.replace("**", "")
    remove_zh_str = _remove_brackets_and_content(remove_zh_str)
    remove_zh_str = _clean_en_filler_words(remove_zh_str, prefix_last_char)
    return remove_zh_str


# @ray.remote
def format_text(text, text_lang, text_prefix_last_char="", is_auto_lang=False):
    start_time = time.time()
    if is_auto_lang:
        text = _format_zh_or_keep_other_langs(text, text_prefix_last_char)
    else:
        if text_lang != "zh":
            text = _format_en_text(text, text_prefix_last_char)
        else:
            text = _format_zh_text(text, text_prefix_last_char)

    text = (text
            .replace("<unk>", "")
            .replace("  ", " ")
            .replace("\n", " ")
            .replace("...", "")
            .replace("…", "")
            )

    text = re.sub(r"\s+", " ", text)
    logger.info(f"format_text end cost: {time.time() - start_time:.3f}")
    return text


def post_process(text: str, lang: str = 'zh'):
    if "�" in text:
        text = text.replace("�", "")

    if lang != 'en':
        return text
    # 需确保使用word level hold-n时才可以使用，token level hold-n会拆分英文单词
    if text and ((text[0] in string.ascii_letters) or text[0].isspace()):
        # 英文句首没有标点的情况下，需要添加空格，避免与上文粘黏
        text = " " + text.lstrip()
    return text
