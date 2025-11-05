import torch
import torchaudio
import torchaudio.transforms as T

def extract_melspectrogram(
    waveform: torch.Tensor,
    sr: int,
    target_sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 400,
    hop_length: int = 160,
    seconds: float = 3.0,
) -> torch.Tensor:

    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_len = int(target_sr * seconds)
    if waveform.shape[-1] < target_len:
        pad = target_len - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif waveform.shape[-1] > target_len:
        start = (waveform.shape[-1] - target_len) // 2
        waveform = waveform[..., start:start + target_len]

    melspec = T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )(waveform)

    logmel = T.AmplitudeToDB(stype="power")(melspec)

    mean = logmel.mean(dim=-1, keepdim=True)
    std = logmel.std(dim=-1, keepdim=True).clamp_min(1e-6)
    logmel = (logmel - mean) / std

    return logmel.to(torch.float32)


_hubert_bundle = torchaudio.pipelines.HUBERT_BASE
_hubert_model = _hubert_bundle.get_model()
_hubert_model.eval()  

def extract_hubert(
    waveform: torch.Tensor,
    sr: int,
    target_sr: int = 16000,
    seconds: float = 3.0,
) -> torch.Tensor:

    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_len = int(target_sr * seconds)
    if waveform.shape[-1] < target_len:
        pad = target_len - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif waveform.shape[-1] > target_len:
        start = (waveform.shape[-1] - target_len) // 2
        waveform = waveform[..., start:start + target_len]


    with torch.inference_mode():
        features, _ = _hubert_model(waveform)  

    return features.squeeze(0)
