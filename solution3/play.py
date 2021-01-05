"""
实时预测

数据集：在原始音频数据的基础上采样音频片段，每个样本为一小段音频，使用帧序列表示该段音频，每一帧表示25ms。
        每个样本对样的输出为最后一帧中各个物种声音出现的概率
"""

import torch
from torch import nn
import torchvision.models as models
from utils import vggish_melspectrogram, vggish_wave_data_mel
import soundfile as sf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import label_ranking_average_precision_score as skllrap
import torchlibrosa as tl


data_info = pd.read_csv("../data/train_all.csv")
gs = data_info.groupby("recording_id")
frame_length = 0.025
piece_length = frame_length * 8
stride = 0.1
num_class = 24


def sample_one_audio(file, info: pd.DataFrame):
    """
    采样方法：在每个t_min前的2.5秒到t_max后的2.5秒，间隔0.1秒，截取0.2秒
    :param file:
    :param info:
    :return:
    """
    data, sr = sf.read(file, dtype="int16")
    X = []
    Y = []
    for ridx in range(len(info)):
        row = info.iloc[ridx]
        positive = row['positive']
        sp_id = row["species_id"]
        t_min = row["t_min"]
        t_max = row["t_max"]
        left = max(t_min-2.5, 0)
        right = min(t_max+2.5, 60)
        while True:
            s = left
            start = int(sr * s)
            end = int(sr * (s+piece_length))
            piece = data[start: end]
            piece_mel = vggish_wave_data_mel(piece, sr)
            X.append(piece_mel)
            label = np.zeros(num_class)
            if s+piece_length < t_min:
                Y.append(label)
            elif s+piece_length <= t_max:
                label[sp_id] = 1 if positive == 1 else -1
                Y.append(label)
            else:
                Y.append(label)
            left += stride
            if s+piece_length > right:
                break

    return X, Y


def generate_dataset():
    root = "../data/train/"
    features_path = "./Features."
    audios = os.listdir(root)
    Features = []
    Labels = []

    for idx, audio in enumerate(audios):
        path = os.path.join(root, audio)
        g = gs.get_group(audio.split(".")[0])
        x, y = sample_one_audio(path, g)
        Features.extend(x)
        Labels.extend(y)

        print(f"already processed {idx+1} / {len(audios)}, samples num: {len(x)}, total num: {len(Features)}")

    Features = np.array(Features)
    Labels = np.array(Labels)
    np.savez("dataset", Features, Labels)


class AudioClassifier(nn.Module):
    def __init__(self, classes_num):
        super().__init__()

        self.backbone = models.resnet50(pretrained=True, progress=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.max_pool = AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)

        self.linear1 = nn.Linear(self.backbone.fc.out_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(256, classes_num)

        # self.sigmoid = nn.Sigmoid()

    #         self.fc = Linear(encoder_params[encoder]['features'], classes_num)

    def forward(self, x):
        x = self.backbone(x)
        #         print(x.shape)
        #         x = self.avg_pool(x).flatten(1)

        x = self.dropout(x)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc(x)

        # x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    model_pth = "./model08.pt"
    model = torch.load(model_pth)
    model.eval()
    data = np.load("./species_x_y (1).npz", allow_pickle=True)
    X, Y = data['x'], data['y']
    X = torch.tensor(X, dtype=torch.float32)
    batch = 64
    i = 0
    preds = []
    with torch.no_grad():
        while True:
            start = i * batch
            end = min(X.shape[0], (i + 1) * batch)
            tmp_preds = model(X[start:end])
            preds.extend(tmp_preds.data.numpy())
            # print(tmp_preds)
            if end == X.shape[0]:
                break
            i += 1

    print(preds)

    loss_function = nn.BCEWithLogitsLoss()
    preds = torch.tensor(preds, dtype=torch.float32)
    Y_tensor = torch.tensor(Y)
    print(loss_function(Y_tensor, preds))

    preds = torch.sigmoid(preds)
    preds = preds.data.numpy()
    print(preds.shape)
    print(skllrap(Y, preds))

    np.savez(f"./species_preds_labels{model_pth.split('./')[-1].split('.')[0]}", preds=preds, labels=Y)

    # generate_dataset()
    # file = "../data/train/003bec244.flac"
    # g = gs.get_group(file.split("/")[-1].split(".")[0])
    # X, Y = sample_one_audio(file, g)
    # print(X)









