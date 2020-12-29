"""
将每个音频转化为一个频谱图，直接用于多标签分类
"""
import librosa
import librosa.display
import matplotlib.pyplot as pyplot
import pandas as pd
import os
import numpy as np

CLASS_NUM = 24

def preprocess():
    train_all = pd.read_csv("../data/train_all.csv")

    return train_all


def convert(train: pd.DataFrame):
    recording_dir = "../data/train/"
    save_dir = "./data/spectrograms/"
    sampling_rate = 48000
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # recordings = train['recording_id'].unique().tolist()
    g = train.groupby("recording_id")
    i = 0
    for record, group in g:
        filename = record + '.png'  # 频谱图的文件名
        audio_file_path = recording_dir + record + ".flac"  # 音频文件路径
        audio_data, sampling_rate = librosa.load(audio_file_path, sr=sampling_rate)
        if os.path.exists(save_dir + filename):
            print(f"{save_dir + filename} already exists, skip generating it ... {i} / {len(g)}")
            i += 1
            continue

        S = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sampling_rate,
            n_fft=2048,
            hop_length=512,
            win_length=1024)
        dpi = 100
        fig = pyplot.figure(num=None, figsize=(300 / dpi, 300 / dpi), dpi=dpi)
        pyplot.subplot(222)
        ax = pyplot.axes()
        ax.set_axis_off()
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        pyplot.savefig(save_dir + filename, bbox_inches='tight',
                       transparent=True, pad_inches=0.0)
        pyplot.close()
        print(f"generating {save_dir + filename} finished ... {i+1} / {len(g)}")
        i += 1


def gen_target(train: pd.DataFrame):
    g = train.groupby("recording_id")
    targets = []
    for record, group in g:
        tar = np.zeros(CLASS_NUM)

        for idx in range(len(group)):
            item = group.iloc[idx]
            species_id = item['species_id']
            if 1 == item['positive']:
                tar[species_id] = 1
            else:
                tar[species_id] = -1

        targets.append(tar)

    df = pd.DataFrame()


    return targets


if __name__ == "__main__":
    train = preprocess()
    # convert(train)
    targets = gen_target(train)
    print(targets)