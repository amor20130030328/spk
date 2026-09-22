from meeting_processor import processor
_ITN_CACHE_DIR = "/opt/huawei/hivoice/417742b5-a1f9-4450-aeb2-cdc440d469a3/model/itn"
text = "中华人民共和国成立于一九四九年十月一日 | 1949/10/01"
text = "现在录制的场景是纯线下全真人，呃，录音设备是摩根F录音笔X二品牌手机。"
from itn.chinese.inverse_normalizer import InverseNormalizer
_normalizer = InverseNormalizer(
                cache_dir=_ITN_CACHE_DIR,
                overwrite_cache=False,
                enable_standalone_number=True,
                enable_0_to_9=False,
                enable_million=False,
            )
text = processor.process(text)
out = _normalizer.normalize(text)  # type: ignore[union-attr]
out = out.strip()
print(text,"->",out)
