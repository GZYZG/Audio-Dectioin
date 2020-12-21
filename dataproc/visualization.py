import librosa
import librosa.display
import pandas as pd
import soundfile as sf
import glob
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings(action="ignore")


def display_specshow(file_path, y_axis="log", x_axis="time"):
    """
    显示一段音频的频谱图
    :param file_path: 文件路径
    :param y_axis:
    :param x_axis:
    :return:
    """
    if isinstance(file_path, str):
        file_path = [file_path]

    fig, ax = plt.subplots(nrows=len(file_path), ncols=1, sharex=True)
    for idx, file in enumerate(file_path):
        y, sr = librosa.load(file)

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        tmp_ax = ax[idx] if len(file_path) > 1 else ax
        img = librosa.display.specshow(D, y_axis=y_axis, x_axis=x_axis,
                                   sr=sr, ax=tmp_ax)

        # title = "" or file_path
        tmp_ax.set(title=file)
        tmp_ax.label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()


def display_wave(file_path, duration=60):
    """
    显示一段音频的波形图
    :param file_path: 文件路径
    :param duration: 显示的时长，单位为秒
    :return:
    """
    if isinstance(file_path, str):
        file_path = [file_path]

    fig, ax = plt.subplots(nrows=len(file_path), sharex=True, sharey=True)

    for idx, file in enumerate(file_path):
        y, sr = librosa.load(file, duration=duration)
        tmp_ax = ax[idx] if len(file_path) > 1 else ax
        librosa.display.waveplot(y, sr=sr, ax=tmp_ax)
        tmp_ax.set(title=file)
        tmp_ax.label_outer()

    plt.show()


def extract_data_by_time(data, sr, t_min, t_max):
    """
    提取指定时间段的数据
    :param data: 音频数据
    :param sr: 音频数据的采样频率
    :param t_min: 起始时间
    :param t_max: 终止时间
    :return:
    """
    start = int(sr * t_min)
    end = int(sr * t_max)
    return start, end, data[start:end]


def extract_data_by_id(df: pd.DataFrame, recording_id):
    return df[df["recording_id"] == recording_id]


def annotate_spec(file_path, annotation_info: pd.DataFrame):
    """
    显示标注后的频谱图，正样本被标记为绿色，负样本被标记为蓝色
    :param file_path: 文件路径
    :param annotation_info: 标注信息，来源于train_fp.csv、train_tp.csv
    :return:
    """
    y, sr = librosa.load(file_path)
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    y = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(y, y_axis="log", x_axis="time",
                             sr=sr, ax=ax)
    print(f"file: {file_path}\tnum of info pieces: {len(annotation_info)}")
    for idx in range(len(annotation_info)):
        info = annotation_info.iloc[idx]
        t_min, t_max = info["t_min"], info["t_max"]
        f_min, f_max = info["f_min"], info["f_max"]
        start, end, audio = extract_data_by_time(y, sr, t_min, t_max)
        print(f"info[{idx}]: species_id: {info['species_id']}\tstart: ({start} / {t_min}s)\tend: ({end} / {t_max}s)\t"
              f"f_min: {f_min}Hz  f_max: {f_max}Hz")
        color = "g" if info["positive"] == 1 else "b"
        # ax.plot(np.linspace(t_min, t_max, len(audio)), np.array(audio), color=color, alpha=.5)
        ax.text(t_min, (f_max+f_min)/2, info["species_id"], color="white")
        ax.axvline(t_min, color=color)
        ax.axvline(t_max, color=color)
        ax.add_patch(plt.Rectangle(xy=(t_min, f_min),
                                   width=t_max-t_min, height=f_max-f_min,
                                   color=color, fill=False, linewidth=2))
    ax.set(title=file_path)
    ax.label_outer()
    print()
    plt.show()


def annotate_wave(file_path, annotation_info: pd.DataFrame):
    """
    显示标注后的波形图，正样本被标记为绿色，负样本被标记为红色
    :param file_path: 文件路径
    :param annotation_info: 标注信息，来源于train_fp.csv、train_tp.csv
    :return:
    """
    y, sr = librosa.load(file_path)
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    librosa.display.waveplot(y, sr=sr, ax=ax, alpha=0.2)
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # librosa.display.specshow(D, y_axis="log", x_axis="time",
    #                          sr=sr, ax=ax)
    print(f"file: {file_path}\tnum of info pieces: {len(annotation_info)}")
    for idx in range(len(annotation_info)):
        info = annotation_info.iloc[idx]
        t_min, t_max = info["t_min"], info["t_max"]
        start, end, audio = extract_data_by_time(y, sr, t_min, t_max)
        print(f"info[{idx}]: species_id: {info['species_id']}\tstart: ({start} / {t_min}s)\tend: ({end} / {t_max}s)")
        color = "g" if info["positive"] == 1 else "r"
        ax.plot(np.linspace(t_min, t_max, len(audio)), np.array(audio), color=color, alpha=.5)
        ax.text(t_min, 0.1, info["species_id"], color="k")
        ax.axvline(t_min, color=color)
        ax.axvline(t_max, color=color)
    ax.set(title=file_path)
    ax.label_outer()
    print()
    plt.show()


def annotate_wave_spec(file_path, annotation_info: pd.DataFrame):
    """
    显示标注后的波形图和频谱图，可视化配置参考annotate_wave和annotate_spec
    :param file_path: 文件路径
    :param annotation_info: 标注信息，来源于train_fp.csv、train_tp.csv
    :return:
    """
    y, sr = librosa.load(file_path)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
    librosa.display.waveplot(y, sr=sr, ax=ax[0], alpha=0.2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis="log", x_axis="time",
                             sr=sr, ax=ax[1])
    print(f"file: {file_path}\tnum of info pieces: {len(annotation_info)}")
    for idx in range(len(annotation_info)):
        info = annotation_info.iloc[idx]
        t_min, t_max = info["t_min"], info["t_max"]
        f_min, f_max = info["f_min"], info["f_max"]
        start, end, audio = extract_data_by_time(y, sr, t_min, t_max)
        print(f"info[{idx}]: species_id: {info['species_id']}\tstart: ({start} / {t_min}s)\tend: ({end} / {t_max}s)\t"
              f"f_min: {f_min}Hz  f_max: {f_max}Hz")
        color = "g" if info["positive"] == 1 else "r"
        ax[0].plot(np.linspace(t_min, t_max, len(audio)), np.array(audio), color=color, alpha=.5)
        ax[0].text(t_min, 0.1, info["species_id"], color="k")
        ax[0].axvline(t_min, color=color)
        ax[0].axvline(t_max, color=color)

        color = "g" if info["positive"] == 1 else "b"
        ax[1].text(t_min, (f_max+f_min)/2, info["species_id"], color="white")
        ax[1].axvline(t_min, color=color)
        ax[1].axvline(t_max, color=color)
        ax[1].add_patch(plt.Rectangle(xy=(t_min, f_min),width=t_max - t_min, height=f_max - f_min,
                                      color=color, fill=False, linewidth=2))
    ax[0].set(title=file_path)
    ax[0].label_outer()
    ax[1].label_outer()
    print()
    plt.show()

if __name__ == "__main__":
    base_dir = "../data/train/"

    train_tp = pd.read_csv("../data/train_tp.csv")
    train_tp["positive"] = [1] * len(train_tp)
    train_fp = pd.read_csv("../data/train_fp.csv")
    train_fp["positive"] = [0] * len(train_fp)
    train = pd.concat([train_tp, train_fp], axis=0)
    g = train.groupby("recording_id")
    recordings = train["recording_id"].unique()

    for idx in range(len(recordings)):

        item0 = train.iloc[idx]
        sample = item0["recording_id"]
        sample = base_dir + sample + ".flac"
        t_min = item0["t_min"]
        t_max = item0["t_max"]
        annotate_wave_spec(sample, g.get_group(item0["recording_id"]))

    # all_flac_fs = glob.glob("../data/train/*.flac")
    samples = ['../data/train\\00ad36516.flac', '../data/train\\003b04435.flac', '../data/train\\003bec244.flac',
               '../data/train\\005f1f9a5.flac', '../data/train\\006ab765f.flac', '../data/train\\0072f0839.flac',
               ]  # all_flac_fs[:20]


    print(samples)
    # display_specshow([samples[0], "s0_ns.wav"])

    duration = 60
    # display_wave([samples[0], "s0_ns.wav"], duration=duration)





