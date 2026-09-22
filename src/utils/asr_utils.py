import re


def is_need_tail_space(lang):
    if lang in ["zh", "yue", "ja", "ko", "ar", "th", "ur"]:
        return False
    return True


def is_ends_with_period(text: str) -> bool:
    if not text:
        return False
    english_punctuation = set("?!.")  # 英文标点
    chinese_punctuation = set("。；？！")  # 中文标点
    all_punctuation = english_punctuation.union(chinese_punctuation)
    last_char = text[-1]
    return last_char in all_punctuation


def is_sentence_too_long(text: str) -> bool:
    # 处理空字符串的情况
    if not text:
        return False

    # 判断是否为中文字符串（包含中文字符）
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)

    if has_chinese:
        # 中文字符串：按字符数判断
        return len(text) > 20
    else:
        # 英文字符串：按单词数判断
        # 使用split()分割单词，它会自动处理多个空格
        words = text.split()
        return len(words) > 15


def split_by_punc(s: str, lang="auto"):
    """
    根据指定的标点符号将字符串分为两部分，并应用特定规则。

    规则：
    1. 分割标点：'.', '?', ',', '!', ':' (或中文对应标点)。
    2. 使用句子中从右到左找到的第一个符合条件的标点进行分割。
    3. 分割时，标点符号保留在第一部分的末尾。
    4. '.' 不应是网址或小数的一部分 (例如 "example.com", "3.14")。
    5. ',' 不应是数字中的千位分隔符 (例如 "1,000")。
    6. 如果没有找到合适的分割标点，则返回原始字符串。

    Args:
        s (str): 输入的字符串。
        lang (str, optional): 语言, "en" 或 "zh". 默认为 "en".

    Returns:
        tuple[str, str] | str: 一个包含两部分字符串的元组，或者在没有找到分割点时返回原始字符串。
    """

    if lang == 'auto':
        chinese_char = re.search(r'[\u4e00-\u9fff]', s)
        lang = 'zh' if chinese_char else 'en'

    if lang == "zh":
        punctuations = {'。', '？', '，', '！', '：'}
    else:
        punctuations = {'.', '?', ',', '!', ':'}

    n = len(s)
    split_index = -1

    for i in range(n - 1, -1, -1):
        char = s[i]
        if char in punctuations:
            if char == '.':
                flag1 = i > 0 and i + 1 < n and s[i - 1].isalnum() and s[i + 1].isalnum()
                if flag1:
                    continue
            elif char == ',':
                flag2 = i > 0 and i + 1 < n and s[i - 1].isdigit() and s[i + 1].isdigit()
                if flag2:
                    continue
            split_index = i
            break

    return s, split_index


def sync_split(src_cache, curr_src, dst_cache, curr_dst):
    # # 1. 预处理拼接
    # # 使用 str() 确保拼接安全，防止 None 或非字符串类型混入
    full_src = str(src_cache) + str(curr_src)
    full_dst = str(dst_cache) + str(curr_dst)

    # 2. 分别寻找分割点
    src_text, src_idx = split_by_punc(full_src, lang="auto")
    dst_text, dst_idx = split_by_punc(full_dst, lang="auto")

    # 3. 判断是否满足同步拆分条件
    if src_idx != -1 and dst_idx != -1:
        # === 执行拆分 ===

        # --- 修复开始 ---
        # 强制将索引转换为整数，防止字符串类型的索引导致拼接错误
        # 同时确保 src_text 和 dst_text 是字符串
        src_idx = int(src_idx)
        dst_idx = int(dst_idx)

        if not isinstance(src_text, str):
            src_text = str(src_text)
        if not isinstance(dst_text, str):
            dst_text = str(dst_text)
        # --- 修复结束 ---

        # 原文切分
        src_part = src_text[:src_idx + 1]
        src_remain = src_text[src_idx + 1:].strip()

        # 译文切分
        dst_part = dst_text[:dst_idx + 1]
        dst_remain = dst_text[dst_idx + 1:].strip()

        return src_part, src_remain, dst_part, dst_remain
    else:
        # === 不满足拆分条件 ===
        return "", full_src, "", full_dst


def get_tts_timestamp_direct(data, remain_str, tts_duration_cache):
    """
    直接通过索引获取末尾短语的时间戳
    修改点：使用字符长度对齐来计算 split_index，以兼容中英文
    修复：空列表、越界、负数长度、边界切分等异常
    """
    words_list = data.get('words', [])
    if not words_list:
        return 0, 0

    total_chars_len = sum(len(w['word']) for w in words_list)
    remain_len = len(remain_str)

    if remain_len == 0:
        curr_tts_duration = tts_duration_cache + words_list[-1]['end'] - words_list[0]['start']
        return curr_tts_duration, 0

    remain_len = min(remain_len, total_chars_len)
    split_index_char = total_chars_len - remain_len

    current_len = 0
    split_index = 0
    for i, w in enumerate(words_list):
        word_len = len(w['word'])
        if current_len + word_len > split_index_char:
            split_index = i
            break
        current_len += word_len
    else:
        split_index = len(words_list)

    split_index = max(0, min(split_index, len(words_list)))

    if split_index <= 0:
        return tts_duration_cache, 0

    p1_start = words_list[0]['start']
    p1_end = words_list[split_index - 1]['end']
    curr_tts_duration = tts_duration_cache + p1_end - p1_start

    p2_start = words_list[split_index]['start']
    p2_end = words_list[-1]['end']
    tts_duration_cache = p2_end - p2_start

    return curr_tts_duration, tts_duration_cache
