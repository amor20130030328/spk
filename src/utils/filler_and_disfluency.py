import re
import string

import jieba

from src.configs.common_punctuation import simple_punctuation, IGNORE_WORDS

# -------- previous filter -------------
fillers = [
    '这个', '那么', '就是说', '就是', '那个', '然后', '但是', '非常', '这是',
    '那', '这', '吧', '哦', '了', '吗', '嘛', '嘿', '啦',
    '喏', '喂', '喲', '喳', '啍', '噢', '呀', '一个', '这种', '哎呀']

filler_char = [
    '啊', '嗯', '唉', '哎', '哦', '喔', '额', '嘛', '嘿', '啦', '喏',
    '喲', '喳', '啍', '噢', '呀', '诶', '哈', "呐"
]

filler_string = ['是吧', '对吧', '呃', '⚡', '❓', '╱╱╱╱╱', '❌', '_', '®', '±']

filler_string_with_punc = ['是吧？', '对吧？']


def _rule_disfluency(text):
    for word in fillers:
        text = re.sub(f"({word}[,.，。]?)+", "\g<1>", text)
    return text


def _check_text_rm_zh(text, text_rm_zh):
    if len(text) <= 30:
        return False
    return 'a' <= text[1].lower() <= 'z' and 'a' <= text[-2].lower() <= 'z' and len(text) - len(text_rm_zh) <= 3


def _repeat_single_char(word, words, i):
    if i + 1 >= len(words) or len(word) != 1:
        return False
    return '\u4e00' <= word <= '\u9fff' and words[i + 1] and word == words[i + 1][0]


def _rule_filler_char(text):
    text = text.replace('哈喽', 'Hello')
    for word in filler_string_with_punc:
        text = text.replace(word, '。')
    for word in filler_string:
        text = text.replace(word, '')
    # 去掉上一句末尾的儿化音
    if text and text[0] in ['儿', '了']:
        text = text[1:]
    # 去掉英文里包含的中文
    text_rm_zh = re.sub(r'[\u4e00-\u9fff]', '', text)
    if _check_text_rm_zh(text, text_rm_zh):
        text = text_rm_zh
    words = [word for word in jieba.cut(text, cut_all=False)]
    new_words = []
    for i, word in enumerate(words):
        if word in filler_char:
            # 如果口水词后面跟着数字，不要轻易替换为''，先换为空格。
            if i + 1 < len(words) \
                    and words[i + 1] \
                    and words[i + 1][0] in string.digits:
                new_words.append(' ')
            continue
        # 单字重复的场景：我我们会很喜欢。
        if _repeat_single_char(word, words, i):
            continue
        # 重复词场景：根据根据我们的经验
        if i and word == words[i - 1] and len(word) > 1:
            continue
        new_words.append(word)
    return ''.join(new_words).replace('�', '').replace('⚠️', '').replace('⚠', '')


# -------- en filter -------------
EN_FILTER_WORDS = [
    "um", "uh", "ah", "hmm", "er", "mhm", "you know"
]


def _clean_en_filler_words(text, prefix_last_char: str | None = ""):
    # --- 步骤 1: 保护缩写词 (Masking) ---
    # 将 "Mr." 替换为 "{ABBR_0}", "etc." 替换为 "{ABBR_1}" 等
    # 这样正则处理标点时，就不会把它们当成句号了

    abbr_map = {}
    masked_text = text
    is_head_space = False
    is_head_punc = False
    if not text.strip():
        return text
    if text and text[0] == " ":
        is_head_space = True
    elif text and text[0] in simple_punctuation:
        is_head_punc = True

    # 按长度降序排列，防止 "No." 匹配到 "No" 而漏掉点，或者包含关系的词匹配错误
    # 实际上这里主要是为了确保准确匹配
    sorted_abbrs = sorted(IGNORE_WORDS, key=len, reverse=True)

    for i, abbr in enumerate(sorted_abbrs):
        # 生成一个唯一的占位符，不包含任何标点符号
        placeholder = f"{{{{ABBR_{i}}}}}"
        abbr_map[placeholder] = abbr
        # 使用 re.escape 确保缩写词中的点号被当作普通字符
        # \b 确保边界（但要注意 . 也是边界，所以对于以点结尾的词，直接匹配即可）
        # 这里直接替换文本中的缩写
        pattern = re.escape(abbr)
        masked_text = re.sub(pattern, placeholder, masked_text)

    # --- 步骤 2: 执行核心清理逻辑---

    # 2.1 删除口水词
    pattern = r'\b(?:' + '|'.join(map(re.escape, EN_FILTER_WORDS)) + r')\b'
    cleaned = re.sub(pattern, '', masked_text, flags=re.IGNORECASE)

    # 2.2 压缩空格
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # 2.3 修复标点前的空格
    cleaned = re.sub(r'\s+([,.?!:;])', r'\1', cleaned)

    # 2.4 处理标点冲突
    cleaned = re.sub(r'([.?!])(?:\s*[.?!,;])+', r'\1', cleaned)
    cleaned = re.sub(r'[,;](?:\s*[,;])*\s*([.?!,;])', r'\1', cleaned)

    # 2.5 智能大小写修复
    # 仅修复真正句号(.?!)后面的单词
    def capitalize_match(match):
        return match.group().upper()

    cleaned = re.sub(
        r'(?<=[.?!]["\']\s)[a-z]|(?<=[.?!]\s)[a-z]', capitalize_match, cleaned)

    # --- 步骤 3: 还原缩写词 (Unmasking) ---

    for placeholder, original_abbr in abbr_map.items():
        cleaned = cleaned.replace(placeholder, original_abbr)

    cleaned = cleaned.strip()
    if not is_head_punc and cleaned and cleaned[0] in simple_punctuation:
        # 原版没有句首标点，处理后新出了句首标点
        if prefix_last_char and prefix_last_char[-1] not in simple_punctuation:
            # 上一句句尾没有标点，则保留句首标点
            pass
        else:
            # 场景1：上一句句尾有标点
            # 场景2：看不到上一句
            cleaned = cleaned[1:].strip()
    return f" {cleaned}" if is_head_space and cleaned and cleaned[0] not in simple_punctuation else cleaned


HALLUCINATION_LIST = [
    "在1998年，他被任命为美国国家",
    "他于1998年",
    "在1980年代，他与妻子和",
    "在1990年代，他与妻子和",
    "在1999年，他被选为",
]

EMPTY_PREFIX_HALLUCINATION_LIST = [
    "该片由陈可辛执导",
    "该片由约翰·麦克蒂尔",
    "我叫王小美",
    "本剧于2015年10月19日",
    "他于一九八九年逝世",
    "该岛是印度洋上的一个火山岛",
    "该片由华纳兄弟影片公司发行",
    "该区是伦敦金融城的一部分",
    "该片于2013年10月18日在美国上映",
    "该岛是该国的第二大岛",
    "在2014年，他与前队友大卫·比利亚",
    "在1990年代，他开始在电视上",
    "他把他的手放在我的头上",
    "Yes, yes, anyway he was a good man",
    "The first time I saw him, I thought he was a very good person",
    "The first is the most important",
    "not going to do that",
    "I'm not going to go",
    "他被指控在2008年1月29日于伦敦的家中被谋杀",
    "他于1998年去世",
    "他现在是自由球员",
    "在2013年，他被任命为美国"
]


def filter_hallucination(text: str, is_first=False) -> str:
    clear_text = text.strip()
    for prefix in HALLUCINATION_LIST:
        if clear_text.lower().startswith(prefix.lower()):
            return ""
    if is_first:
        for prefix in EMPTY_PREFIX_HALLUCINATION_LIST:
            if clear_text.lower().startswith(prefix.lower()):
                return ""
    return text
