# 中文标点符号集合（含全角符号）
chinese_punctuation = '''＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'''

# 英文标点符号集合（含半角符号）
english_punctuation = '''!"#$%&’()*+,-./:;<=>?@[]^_`{|}~'''
all_punctuation = chinese_punctuation + english_punctuation

simple_punctuation = """,.?!;:、，。？！：；"""


ZH2EN_PUNC_MAP = {
    '。': '.',
    '，': ',',
    '？': '?',
    '、': ',',
    '！': '!',
}

EN2ZH_PUNC_MAP = {
    '...': '。',
    # '.': '。', # 通常没有出现中文伴随英文句号的场景
    ',': '，',
    '?': '？',
    '!': '！',
}

IGNORE_WORDS = [
    "etc.", "Mr.", "Mrs.", "Dr.", "Prof.",
    "U.S.", "A.M.", "P.M.", "e.g.", "i.e.",
    "Inc.", "No.", "Gov.", "Jr.", "Sr.",
    "St.", "Ave.", "Inc.", "Ltd.", "Co.",
    "U.K.", "E.U.",
]

PUNCS = {
    "zh": ["？", "！", "。", "，", "、"],
    "en": [".", "?", "!", ","]
}
