import re
from typing import List

import jieba
import ray
from src.configs.common_punctuation import simple_punctuation, IGNORE_WORDS, all_punctuation


def _split_hold_n_with_tokenizer(src, n=3, tokenizer=None, is_split_as_word=False):
    if not src:
        return src, ''

    head_is_space = False
    if src[0] == " ":
        head_is_space = True

    tok_ids = tokenizer(src)["input_ids"]
    hold_n_txt = tokenizer.decode(tok_ids[:-n])
    partial = tokenizer.decode(tok_ids[-n:])
    while len(tok_ids) >= n:
        hold_n_txt = tokenizer.decode(tok_ids[:-n])
        partial = tokenizer.decode(tok_ids[-n:])
        if not is_split_as_word:
            break
        # 要求切断点必须是完整单词
        # bad case: infrastructures -> infra structures, 而ctc的分词为会将单独的 infra 分词为 ▁in f ra
        if partial and partial[0] != " " and partial[0] not in all_punctuation:
            n += 1
        else:
            break

    # 有概率出现句首空格被移除
    if head_is_space and hold_n_txt and hold_n_txt[0] != " ":
        hold_n_txt = " " + hold_n_txt

    return hold_n_txt, partial


def _find_common_split_and_split_final(
    list1: List[str],
    list2: List[str],
    N: int
):
    original_string = "".join(list1)
    cumulative_len1 = 0
    split_indices1 = {0}
    for item in list1:
        cumulative_len1 += len(item)
        split_indices1.add(cumulative_len1)

    cumulative_len2 = 0
    split_indices2 = {0}
    for item in list2:
        cumulative_len2 += len(item)
        split_indices2.add(cumulative_len2)

    common_indices = sorted(list(split_indices1.intersection(split_indices2)))

    # 从右向左遍历共同分割位置，找到第一个满足 N 条件的
    for split_index in reversed(common_indices):

        # 如果当前指针对应的元素末尾位置大于 split_index，则向前移动指针

        # L1 剩余元素个数 (基于 split_index)
        count1 = 0
        current_len = len(original_string)
        # 从后向前计算，直到累积长度 <= split_index
        for i in range(len(list1) - 1, -1, -1):
            if current_len > split_index:
                count1 += 1
            current_len -= len(list1[i])

        # L2 剩余元素个数 (基于 split_index)
        count2 = 0
        current_len = len(original_string)
        for i in range(len(list2) - 1, -1, -1):
            if current_len > split_index:
                count2 += 1
            current_len -= len(list2[i])

        # 检查 N 条件
        if count1 >= N or count2 >= N:
            # 找到最右侧满足条件的分割点
            part1 = original_string[:split_index]
            part2 = original_string[split_index:]
            return part1, part2

    # 如果没有找到满足 N 条件的共同分割点，返回第一个共同分割点 (索引 0) 的结果
    first_split_index = common_indices[0]
    return original_string[:first_split_index], original_string[first_split_index:]


def _split_hold_n_with_jieba_and_tokenizer(src, n=3, tokenizer=None):
    if not src:
        return src, ''

    src_split = [src]

    partial = ""
    hold_n_remain = n

    jieba_words = [w for w in jieba.cut(src_split[-1])]
    while len(jieba_words) < hold_n_remain:
        # 最后一个文本片段长度不足
        partial = src_split[-1] + partial
        src_split = src_split[:-1]

        if len(src_split) > 1:
            partial = src_split[-1] + partial
            src_split = src_split[:-1]
            hold_n_remain -= (len(jieba_words) + 1)
            jieba_words = [w for w in jieba.cut(src_split[-1])]
        else:
            return "", src

    if hold_n_remain < 1:
        return src[:-len(partial)], partial

    tok_ids = tokenizer(src_split[-1])["input_ids"]
    token_words = [tokenizer.decode(tok_id) for tok_id in tok_ids]

    _, tail = _find_common_split_and_split_final(
        jieba_words, token_words, hold_n_remain)

    partial = tail + partial
    return src[:-len(partial)], partial


def split_hold_n(src: str, lang: str = 'zh', tokenizer=None, n=None):
    if not src or not tokenizer or n == 0:
        return src, ''
    lang = lang or 'zh'

    n = n or 3

    if lang == 'zh':
        return _split_hold_n_with_jieba_and_tokenizer(src, n, tokenizer)
    is_split_as_word = lang in ["en"]
    return _split_hold_n_with_tokenizer(src, n, tokenizer, is_split_as_word)


# 定义结束标点
END_PUNCTUATION = set(['.', '?', '!', '。', '？', '！', ';', '；'])

EN_END_PUNCTUATION = set(['.', '?', '!', ';'])


def split_text_before_sent(text: str):
    if not text:
        return [text]

    # --- 步骤 1: 处理句首标点 (至多1个) ---
    head = None
    body = text

    # 只要第一个字符是标点，就切分出来
    if text[0] in simple_punctuation:
        head = text[0]
        body = text[1:]

    # 如果切分后主体为空，直接返回
    if not body:
        return [head]

    # --- 步骤 2: 标记主体中所有需要忽略的标点索引 ---
    # 构建正则：(?<![a-zA-Z0-9]) 确保前面不是字母数字，避免 No. 匹配 Volcano.
    # 使用 re.IGNORECASE 增加容错性
    escaped_words = [re.escape(w) for w in IGNORE_WORDS]
    # 按长度降序排列，确保 U.S. 优先于 U. (如果有的话)
    escaped_words.sort(key=len, reverse=True)
    pattern_str = r'(?<![a-zA-Z0-9])(?:' + '|'.join(escaped_words) + r')'
    pattern = re.compile(pattern_str, re.IGNORECASE)

    ignored_indices = set()
    for match in pattern.finditer(body):
        # 将匹配到的单词范围内的所有标点都加入忽略集合
        for i in range(match.start(), match.end()):
            if body[i] in END_PUNCTUATION:
                ignored_indices.add(i)

    # --- 步骤 3: 寻找最后一个有效的切分点 ---
    split_index = -1

    # 从后往前遍历 body
    for i in range(len(body) - 1, -1, -1):
        char = body[i]
        if char in EN_END_PUNCTUATION:
            # 1. 如果是忽略词中的点，跳过
            if i in ignored_indices:
                continue

            # 2. 检查标点后是否有内容
            content_after = body[i+1:].strip()
            if not content_after:
                continue

            # 3. 如果标点后面不是空格字符，跳过
            char_after = body[i+1]
            if char_after != " ":
                continue

            # 找到符合条件的最后一个点，记录位置并停止寻找
            split_index = i
            break

        elif char in END_PUNCTUATION:
            # 1. 检查标点后是否有内容
            content_after = body[i+1:].strip()
            if not content_after:
                continue

            # 找到符合条件的最后一个点，记录位置并停止寻找
            split_index = i
            break

    # --- 步骤 4: 组装结果 ---
    result = []
    if head:
        result.append(head)

    if split_index != -1:
        # 切分点及其之前的内容
        result.append(body[:split_index+1])
        # 切分点之后的内容
        result.append(body[split_index+1:])
    else:
        # 没有找到有效切分点，整体返回
        result.append(body)

    return result


def update_text_queue_with_split(
    tokenizer,
    text,
    text_queue,
    audio_queue,
    wave_duration
):
    # 如果队列前面有多个空白，则把新出的结果均匀地更新在前面的队列空白中
    # Calculate N as number of trailing empty strings in text_queue
    n = 0
    for i in range(len(text_queue) - 1, -1, -1):
        if text_queue[i] == "":
            n += 1
        else:
            break
    if n == 0 or text == "":
        return text

    # Get audio durations from self.audio_queue
    audio_durations = []
    for i in range(1, n + 1):
        # 这里是从最后一个prefix开始的
        audio_data = audio_queue[-i]
        duration = len(audio_data) * 1.0 / 16000
        audio_durations.append(duration)

    total_audio_duration = sum(audio_durations)
    total_duration = total_audio_duration + wave_duration
    if total_duration <= 0:
        return text

    # Tokenize the text
    tokens = tokenizer(text)["input_ids"]
    total_tokens = len(tokens)

    # Split into parts
    parts = []
    start_idx = 0
    # 需要将audio_durations按prefix顺序来排
    for i, dur in enumerate(list(reversed(audio_durations)) + [wave_duration]):
        ratio = dur / total_duration
        part_tokens = round(ratio * total_tokens)
        end_idx = start_idx + part_tokens

        # 最后一个音频（新音频）
        if i == len(audio_durations):
            end_idx = total_tokens

        part_text = tokenizer.decode(tokens[start_idx:end_idx])
        parts.append(part_text)
        start_idx = end_idx

    # Replace last N elements
    for i in range(n):
        if i < len(parts):
            text_queue[-n + i] = parts[i]
        else:
            text_queue[-n + i] = ""

    # 返回 the (N+1)th part
    if len(parts) > n:
        return parts[n]
    else:
        return ""


def split_by_last_punctuation(
    text: str,
    chinese_punct=r"[，。！？；：]",
    english_punct=r"[,.!?;:](?=\s)"
):
    """
    将一个中文、英文或中英混合字符串按最后一个标点拆分。
    - 中文标点直接识别。
    - 英文标点必须后面跟一个空格才算有效。
    - 如果标点属于 IGNORE_WORDS 的结尾，则不能作为拆分点。
    标点符号包含在前半部分。
    如果没有标点，则返回 (原字符串, "")。
    """
    punct_pattern = f"{chinese_punct}|{english_punct}"

    matches = list(re.finditer(punct_pattern, text))
    if not matches:
        return text, ""

    # 从最后一个标点开始往前找合格的拆分点
    for match in reversed(matches):
        idx = match.end()
        prefix = text[:idx]

        # 检查是否以 IGNORE_WORDS 结尾
        if any(prefix.endswith(word) for word in IGNORE_WORDS):
            continue  # 跳过这个标点

        return text[:idx], text[idx:]

    # 如果所有标点都在 IGNORE_WORDS 中，则不拆分
    return text, ""
