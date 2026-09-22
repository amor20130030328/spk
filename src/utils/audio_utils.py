import torch

def compute_rms(audio_chunk):
    if len(audio_chunk) == 0:
        return 0.0
    return torch.sqrt(torch.mean(audio_chunk.float() ** 2)).item()