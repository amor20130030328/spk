
from src.vpr.feature.core import load_hyperpyyaml


def read_hyperyaml(path):
    with open(path, 'r',  encoding='utf-8') as f:
        data = load_hyperpyyaml(f.read())
        sample_rate = data['sample_rate']
        dur_range = data['dur_range']
        data = data['modules']['spk_model'].front.feature_extractor
    return data, sample_rate, dur_range






