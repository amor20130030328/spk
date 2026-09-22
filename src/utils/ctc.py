import re
from typing import List, Any, Tuple

import jieba

from src.configs.common_punctuation import all_punctuation, simple_punctuation
from src.utils.split import split_hold_n


# -------- hold-n ctc -------------


def _preprocess_text(text: str) -> str:
    """
    执行文本规范化：移除标点符号、转换为小写，并清理空格。

    Args:
        text: 待处理的原始文本。

    Returns:
        清理后的文本字符串。
    """
    # 1. 转换为小写
    text = text.strip().lower()

    # 2. 移除标点符号 (支持中英文)
    text = text.translate(str.maketrans('', '', all_punctuation))

    # 3. 清理多余的空格，并去除前后空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def _find_lcs_contiguous(list1: List[Any], list2: List[Any]) -> Tuple[List[Any], int]:
    """
    查找 list1 和 list2 之间最长的公共连续子列表（LCS Contiguous），
    并返回该子列表及其在 list1 中的起始索引。

    Args:
        list1: 第一个分词列表（后N个词）。
        list2: 第二个分词列表（后N个词）。

    Returns:
        (common_list, start_index): 最长公共子列表和它在 list1 中的起始索引。
                                     如果未找到公共部分，返回 ([], -1)。
    """
    max_len = 0
    end_index_list1 = -1  # 最长匹配在 list1 中结束的索引

    m, n = len(list1), len(list2)
    # 使用动态规划（DP）数组来存储当前匹配长度，这里使用1D数组优化空间
    # dp[j] 存储 list1[i] 结束时与 list2[j] 结束时匹配的长度
    dp = [0] * (n + 1)

    # 遍历 list1
    for i in range(m):
        # 必须使用一个新的DP数组或倒序遍历，以确保当前 i 依赖的是 i-1 的结果
        new_dp = [0] * (n + 1)
        # 遍历 list2
        for j in range(n):
            if list1[i] == list2[j]:
                # 如果匹配，则当前长度等于上一个匹配长度 + 1
                current_len = dp[j] + 1
                new_dp[j + 1] = current_len

                # 检查是否是新的最长公共子列表
                if current_len > max_len:
                    max_len = current_len
                    end_index_list1 = i
            # 如果不匹配，new_dp[j + 1] 保持为 0 (无需显式设置，因为已初始化)

        # 更新 dp 数组，用于下一轮 list1 的迭代
        dp = new_dp

    common_list: List[Any] = []
    start_index_list1 = -1

    if max_len > 0:
        # 根据最长匹配的长度和结束位置，计算在 list1 中的起始位置
        start_index_list1 = end_index_list1 - max_len + 1
        # 提取最长公共子列表
        common_list = list1[start_index_list1:end_index_list1 + 1]

    return common_list, start_index_list1


END_SINGLE_WORDS = ["了", "的", "说", "吗", "都", "你", "他", "吧"]


def _move_zh_last_word(last_final, final_text, partial_text, tokenizer):
    is_move_last_word = False
    final = final_text
    partial = partial_text

    # ('我敢肯定，弹幕的兄弟', '们。')
    if len(last_final) == 1 and last_final not in END_SINGLE_WORDS:
        is_move_last_word = True
    # ('Max的配置是16加', '512。')
    if last_final.isnumeric():
        is_move_last_word = True

    if is_move_last_word:
        final, final_tail = split_hold_n(final, "zh", tokenizer, 1)
        partial = final_tail + partial
    return final, partial


def _split_list_by_subsequence(input1: List[Any], input2: List[Any]):
    """
    将 input1 分割成一个前缀和一个尽可能短的后缀 (return2)，
    该后缀必须包含 input2 作为子序列，并且保持 input2 中元素的顺序。

    Args:
        input1: 要分割的主列表。
        input2: 所需的子序列模式。

    Returns:
        一个元组，包含前缀 (return1) 和后缀 (return2)。
    """
    n1 = len(input1)
    n2 = len(input2)

    # pattern_index 指向我们当前正在 input2 中搜索的元素。
    # 我们从 input2 的末尾开始向后移动。
    pattern_index = n2 - 1

    # 默认分割索引是 n1，这意味着整个列表都是后缀
    # (仅在 input2 为空列表时发生，但根据要求，input2 不为空)。
    split_index = n1

    # 1. 从 input1 向后迭代
    for i in range(n1 - 1, -1, -1):
        # 2. 检查 input2 中是否还有元素需要匹配
        if pattern_index >= 0:
            # 3. 检查是否匹配
            if input1[i] == input2[pattern_index]:
                # 找到该元素！将模式指针向后移动一位（向前搜索）。
                pattern_index -= 1

        # 4. 检查是否已找到完整的 input2 子序列
        if pattern_index < 0:
            # 如果 pattern_index 为 -1，表示我们找到了 input2[0]。
            # 当前索引 'i' 就是最小后缀的起始点。
            split_index = i
            break

    # 注意: 根据要求，input2 中的元素保证存在于 input1 中，
    # 因此 pattern_index 应该总是会小于 0。

    # 5. 根据计算出的索引分割列表
    return1 = input1[:split_index]
    return2 = input1[split_index:]

    return return1, return2


def split_text_by_ctc_text(
        src: str = "",
        prefix: str = "",
        ctc_text: str = "",
        lang: str = "zh",
        tokenizer: Any = None,
        max_tail: int = 8,
):
    if not src or not tokenizer or max_tail == 0:
        return True, src, ""

    text = src

    # 预处理
    cleaned_src = _preprocess_text(text)
    cleaned_prefix = _preprocess_text(prefix)
    cleaned_ctc_text = _preprocess_text(ctc_text)

    # 使用 tokenizer 对文本进行分词
    try:
        src_ids = tokenizer(cleaned_src)["input_ids"]
        asr_ids = tokenizer(cleaned_prefix)["input_ids"]
        ctc_ids = tokenizer(cleaned_ctc_text)["input_ids"]

        # 取出的个数不超过src的范围
        N = min(len(src_ids), max_tail)
        asr_ids.extend(src_ids)
    except Exception as _:
        return False, text, ""  # 发生错误时fallback

    # 各取出两个分词列表的后 N 个词，如果列表长度不足 N，则取出全部
    list_asr = asr_ids[-N:]
    list_ctc = ctc_ids[-N:]

    # 得到 list_asr、list_ctc 两个列表的"最长公共连续子列表" common_list
    common_list, start_index_list1 = _find_lcs_contiguous(list_asr, list_ctc)

    # 确认 common_list 在 list_asr 中的位置，并返回剩余元素个数
    if not common_list:
        return False, text, ""  # 如果没有找到公共部分，则fallback

    # common_list 占据的位置是 start_index_list1 到 start_index_list1 + len(common_list) - 1
    # 剩余元素个数 = list_asr 的总长度 - (起始索引 + 公共子列表的长度)
    partial_length = len(list_asr) - \
                     (start_index_list1 + len(common_list))
    if partial_length < 1:
        final = text
        partial = ""

        # 中文默认要hold-1去除单字结尾
        if lang == "zh" and final:
            text_splits = [word for word in jieba.cut(final, cut_all=False)]
            if text_splits[-1] in ["，", "。", "？", "！"]:
                # omni转录结果通常句号结尾
                partial = text_splits[-1]
                text_splits.pop()
                final = final[:-len(partial)]

            # 原始final可能只有一个标点
            if len(text_splits) < 1:
                return True, final, partial

            last_final = text_splits[-1]
            final, partial = _move_zh_last_word(
                last_final, final, partial, tokenizer)

        return True, final, partial

    partial_ids = list_asr[-partial_length:]
    _, partial_ids = _split_list_by_subsequence(
        tokenizer(text.lower())["input_ids"], partial_ids)

    if len(partial_ids) < 1:
        # src带上符号后的token可能与无符号的不一致
        return False, text, ""

    partial = tokenizer.decode(partial_ids).rstrip()
    final = text.rstrip()[:-len(partial)]

    if lang == "zh" and final:
        last_final = [word for word in jieba.cut(final, cut_all=False)][-1]
        final, partial = _move_zh_last_word(
            last_final, final, partial, tokenizer)

    return True, final, partial


omni_additional_punctuation = '''（）()《》+-%@"“”…·'''
for_ctc_punctuation = simple_punctuation + omni_additional_punctuation


def convert_2_ctc_style(text: str):
    if not text:
        return text
    new_str = []
    for ch in text:
        # ctc结果没有标点
        if ch not in for_ctc_punctuation:
            # ctc结果是英文小写
            new_str.append(ch.lower())
    return "".join(new_str)
