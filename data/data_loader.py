import os, re, glob, random
from typing import Tuple, List, Dict
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset
from utils.features import extract_melspectrogram
from transformers import Wav2Vec2FeatureExtractor

class CREMA_D(Dataset):
    EMO_MAP = {"ANG":0, "HAP":1, "NEU":2, "SAD":3}
    EMO_RE  = re.compile(r'_(ANG|HAP|NEU|SAD)_', re.IGNORECASE)

    def __init__(self, root_dir: str, feature_type: str, max_speakers: int = 20,
                 sample_rate: int = 16000, seconds: float = 3.0, n_mels: int = 64):
        voice_dir = os.path.join(root_dir, "AudioWAV")
        all_wavs = [p for p in glob.glob(os.path.join(voice_dir, "*.wav"))
                    if self.EMO_RE.search(os.path.basename(p))]

        spk_to_paths: Dict[str, List[str]] = {}
        for p in all_wavs:
            spk = os.path.basename(p).split("_", 1)[0]
            spk_to_paths.setdefault(spk, []).append(p)
        speakers = sorted(spk_to_paths.keys())[:max_speakers]
        self.voice_paths = [p for spk in speakers for p in spk_to_paths[spk]]
        if not self.voice_paths:
            raise RuntimeError(f"No .wav found under {voice_dir}")

        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.n_mels = n_mels

    def __len__(self): return len(self.voice_paths)

    def _infer_label(self, path: str) -> int:
        emo = re.search(self.EMO_RE, os.path.basename(path)).group(1).upper()
        return self.EMO_MAP[emo]

    def __getitem__(self, idx: int):
        path = self.voice_paths[idx]
        label = self._infer_label(path)
        wav, sr = torchaudio.load(path)  # [C,T]
        if wav.shape[0] > 1:  # mono
            wav = wav.mean(dim=0, keepdim=True)

        if self.feature_type == "hubert":

            return wav.squeeze(0), torch.tensor(label)
        elif self.feature_type == "mel":
        
            mel = extract_melspectrogram(
                wav, sr, target_sr=self.sample_rate, n_mels=self.n_mels, seconds=self.seconds
            )
            return mel, torch.tensor(label)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")


def _speaker_id_from_path(p: str) -> str:
    return os.path.basename(p).split("_", 1)[0]


def _build_splits_by_speaker(paths: List[str], train_ratio=0.8, val_ratio=0.1, seed=42
) -> Tuple[List[int], List[int], List[int]]:
    spk_to_indices: Dict[str, List[int]] = {}
    for idx, p in enumerate(paths):
        spk = _speaker_id_from_path(p)
        spk_to_indices.setdefault(spk, []).append(idx)
    speakers = list(spk_to_indices.keys())
    rng = random.Random(seed); rng.shuffle(speakers)
    n = len(speakers); n_tr = int(n*train_ratio); n_val = int(n*val_ratio)
    tr, va = set(speakers[:n_tr]), set(speakers[n_tr:n_tr+n_val])
    train_idx, val_idx, test_idx = [], [], []
    for spk, idxs in spk_to_indices.items():
        (train_idx if spk in tr else val_idx if spk in va else test_idx).extend(idxs)
    return train_idx, val_idx, test_idx


fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

def collate_fn_hubert(batch, sampling_rate=16000):
    xs, ys = zip(*batch)  
    
    xs = [x.squeeze().numpy() for x in xs]

    inputs = fe(
        xs,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )

    return inputs.input_values, torch.tensor(ys)

def collate_fn_mel(batch):
    xs, ys = zip(*batch)
    ys = torch.tensor(ys)
    M = xs[0].shape[1]
    max_t = max(x.shape[-1] for x in xs)
    padded = []
    for x in xs:
        if x.shape[-1] < max_t:
            pad = torch.zeros(1, M, max_t - x.shape[-1])
            x = torch.cat([x, pad], dim=-1)
        padded.append(x)
    return torch.stack(padded), ys


def get_dataloader_cremad(root: str, batch_size: int, num_workers: int, seed: int,
                          feature_type: str, max_speakers: int, sample_rate: int,
                          seconds: float, n_mels: int):
    base_ds = CREMA_D(root, feature_type, max_speakers, sample_rate, seconds, n_mels)
    train_idx, val_idx, test_idx = _build_splits_by_speaker(base_ds.voice_paths, 0.8, 0.1, seed)
    train_ds, val_ds, test_ds = Subset(base_ds, train_idx), Subset(base_ds, val_idx), Subset(base_ds, test_idx)

    collate = collate_fn_hubert if feature_type == "hubert" else collate_fn_mel
    dl_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **dl_args)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_args)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_args)

    print(f"Found {len(train_ds)} utterances for training.")
    print(f"Found {len(val_ds)} utterances for validating.")
    print(f"Found {len(test_ds)} utterances for testing.")
    return train_loader, val_loader, test_loader
