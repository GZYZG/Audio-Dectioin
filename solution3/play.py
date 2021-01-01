"""
实时预测

数据集：在原始音频数据的基础上采样音频片段，每个样本为一小段音频，使用帧序列表示该段音频，每一帧表示25ms。
        每个样本对样的输出为最后一帧中各个物种声音出现的概率
"""

import torch
from utils import vggish_melspectrogram, vggish_wave_data_mel
import soundfile as sf
import pandas as pd
import numpy as np
import os
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


if __name__ == "__main__":
    generate_dataset()
    file = "../data/train/003bec244.flac"
    g = gs.get_group(file.split("/")[-1].split(".")[0])
    X, Y = sample_one_audio(file, g)
    print(X)









