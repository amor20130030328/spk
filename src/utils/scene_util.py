# 定义场景常量，对应Java中的public static final String
CAR_SCENE = "car"
VOICE_SCENE = "voice"
SUBTITLE_SCENE = "subtitle"
SIMULTANEOUS_SCENE = "simultaneous"


def is_short_voice_scene(scene: str) -> bool:
    """
    是否是短语音场景（对应原Java的isShortVoiceScene方法）
    :param scene: 场景字符串
    :return: 布尔值，是否为短语音场景
    """
    # Python中用lower()实现忽略大小写比较，对应Java的equalsIgnoreCase
    # 增加None判断，避免传入None时抛出AttributeError，提升健壮性
    if scene is None:
        return False
    scene_lower = scene.lower()
    return scene_lower == VOICE_SCENE or scene_lower == CAR_SCENE


def is_long_voice_scene(scene: str) -> bool:
    """
    是否是长语音场景（对应原Java的isLongVoiceScene方法）
    :param scene: 场景字符串
    :return: 布尔值，是否为长语音场景
    """
    # 增加None判断，提升健壮性
    if scene is None:
        return False
    scene_lower = scene.lower()
    return scene_lower == SUBTITLE_SCENE or scene_lower == SIMULTANEOUS_SCENE
