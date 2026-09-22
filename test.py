_ITN_CACHE_DIR = "/model-data/model/ITN-319-master/zh/itn"
text = "中华人民共和国成立于一九四九年"
text = "中华人民共和国成立于一九四九年十月一日"
from itn.chinese.inverse_normalizer import InverseNormalizer
_normalizer = InverseNormalizer(
                cache_dir=_ITN_CACHE_DIR,
                overwrite_cache=False,
                enable_standalone_number=True,
                enable_0_to_9=False,
                enable_million=False,
            )

out = _normalizer.normalize(text)  # type: ignore[union-attr]
out = out.strip()
print(text,"->",out)
