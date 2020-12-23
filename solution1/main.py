import numpy as np
import librosa as lb
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from resnest.torch import resnest50

NUM_CLASSES = 24
SR = 16_000
DURATION = 60
DATA_ROOT = Path("../data")
TRAIN_AUDIO_ROOT = Path("../data/train")
TEST_AUDIO_ROOT = Path("../data/test")


class MelSpecComputer:
    def __init__(self, sr, n_mels, fmin, fmax):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, y):

        melspec = lb.feature.melspectrogram(
            y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
        )

        melspec = lb.power_to_db(melspec).astype(np.float32)
        return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def normalize(image, mean=None, std=None):
    image = image / 255.0
    if mean is not None and std is not None:
        image = (image - mean) / std
    return np.moveaxis(image, 2, 0).astype(np.float32)


def crop_or_pad(y, length, sr, is_train=True):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    elif len(y) > length:
        if not is_train:
            start = 0
        else:
            start = np.random.randint(len(y) - length)

        y = y[start:start + length]

    y = y.astype(np.float32, copy=False)

    return y


class RFCXDataset(Dataset):

    def __init__(self, data, sr, n_mels=128, fmin=0, fmax=None, is_train=False,
                 num_classes=NUM_CLASSES, root=None, duration=DURATION):

        self.data = data

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr // 2

        self.is_train = is_train

        self.num_classes = num_classes
        self.duration = duration
        self.audio_length = self.duration * self.sr

        self.root = root or (TRAIN_AUDIO_ROOT if self.is_train else TEST_AUDIO_ROOT)

        self.wav_transfos = get_wav_transforms() if self.is_train else None

        self.mel_spec_computer = MelSpecComputer(sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)

    def __len__(self):
        return len(self.data)

    def read_index(self, idx, fill_val=1.0, offset=None, use_offset=True):
        d = self.data.iloc[idx]
        record, species = d["recording_id"], d["species_id"]
        try:
            if use_offset and (self.duration < d["duration"] + 1):
                offset = offset or np.random.uniform(1, int(d["duration"] - self.duration))

            y, _ = lb.load(self.root.joinpath(record).with_suffix(".flac").as_posix(),
                           sr=self.sr, duration=self.duration, offset=offset)

            if self.wav_transfos is not None:
                y = self.wav_transfos(y, self.sr)
            y = crop_or_pad(y, self.audio_length, sr=self.sr)
            t = np.zeros(self.num_classes)
            t[species] = fill_val
        except Exception as e:
            #             print(e)
            raise ValueError()  from e
            y = np.zeros(self.audio_length)
            t = np.zeros(self.num_classes)

        return y, t

    def __getitem__(self, idx):

        y, t = self.read_index(idx)

        melspec = self.mel_spec_computer(y)
        image = mono_to_color(melspec)  # 将单通道的melspec转化为3通道的彩色image
        image = normalize(image, mean=None, std=None)

        return image, t


def get_duration(audio_name, root=TEST_AUDIO_ROOT):
    return lb.get_duration(filename=root.joinpath(audio_name).with_suffix(".flac"))


if __name__ == "__main__":

    data = pd.DataFrame({
        "recording_id": [path.stem for path in Path(TEST_AUDIO_ROOT).glob("*.flac")],
    })
    data["species_id"] = [[] for _ in range(len(data))]

    print(data.shape)
    data["duration"] = data["recording_id"].apply(get_duration)
    data.head()
    data["duration"].hist()
    ds = RFCXDataset(data=data, sr=SR, duration=10)
    x, y = ds[0]
    x.shape, y.shape

    TEST_BATCH_SIZE = 40
    TEST_NUM_WORKERS = 2
    test_data = RFCXDataset(data=data, sr=SR)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, num_workers=TEST_NUM_WORKERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = resnest50(pretrained=False)
    net.load_state_dict(torch.hub.load_state_dict_from_url(
                "https://s3.us-west-1.wasabisys.com/resnest/torch/resnest50-528c19ca.pth", model_dir="./", progress=True, check_hash=True))
    # net = resnest50(pretrained=True).to(device)   # 如果本地没有模型则会下载模型
    n_features = net.fc.in_features
    net.fc = torch.nn.Linear(n_features, NUM_CLASSES)
    net = net.to(device)
    net.load_state_dict(torch.load("./rfcx_resnest50.pth", map_location=device))
    net = net.eval()
    net
    preds = []
    net.eval()
    with torch.no_grad():
        for (xb, yb) in tqdm(test_loader):
            xb, yb = xb.to(device), yb.to(device)
            o = net(xb)
            o = torch.sigmoid(o)
            preds.append(o.detach().cpu().numpy())
    preds = np.vstack(preds)
    preds.shape
    sub = pd.DataFrame(preds, columns=[f"s{i}" for i in range(24)])
    sub["recording_id"] = data["recording_id"].values[:len(sub)]
    sub = sub[["recording_id"] + [f"s{i}" for i in range(24)]]
    print(sub.shape)
    sub.head()
    sub.to_csv("submission.csv", index=False)