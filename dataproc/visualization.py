import librosa
import librosa.display
import pandas as pd
import soundfile as sf
import glob
import matplotlib.pyplot as plt
import numpy as np
import warnings
from utils import *
import resampy
import vggish_params

warnings.filterwarnings(action="ignore")


def display_specshow(file_path, y_axis="log", x_axis="time", spec=["normal", "mel"]):
    """
    显示一段音频的频谱图
    :param spec: 谱图的类别，包括常规的时频谱图和mel时频谱图
    :param file_path: 文件路径
    :param y_axis:
    :param x_axis:
    :return:
    """
    if isinstance(file_path, str):
        file_path = [file_path]

    def normal(data, samplerate=None):
        return librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)

    def mel(data, samplerate):
        tmp = librosa.feature.melspectrogram(y=data, sr=samplerate, n_mels=128)
        return librosa.amplitude_to_db(tmp, ref=np.max)

    fig, ax = plt.subplots(nrows=len(spec), ncols=len(file_path), sharex=True, sharey=True)
    ax = np.reshape(ax, (len(spec), len(file_path)))

    for idx, file in enumerate(file_path):
        y, sr = librosa.load(file)
        y = resampy.resample(y, sr, vggish_params.SAMPLE_RATE)
        for jdx, sp in enumerate(spec):
            D = normal(y) if sp == "normal" else mel(y, sr)
            img = librosa.display.specshow(D, y_axis=y_axis, x_axis=x_axis,
                                           sr=sr, ax=ax[jdx][idx])
            ax[jdx][idx].label_outer()
            ax[jdx][idx].set_xlabel(sp)
            print(f"{sp}: {D.shape}")
        ax[0][idx].set(title=file)
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

    multi_col = False
    if isinstance(file_path, str):
        ncols = 1
        file_path = [file_path]
        y, sr = librosa.load(file_path[0])
        ys = [y]
        srs = [sr]
    elif isinstance(file_path, list) and len(file_path) == 2:
        ncols = 2
        multi_col = True
        y, sr = librosa.load(file_path[0])
        y1, sr1 = librosa.load(file_path[1])
        ys = [y, y1]
        srs = [sr, sr1]
    fig, ax = plt.subplots(nrows=2, ncols=ncols, sharex=False, sharey=True)

    for c in range(ncols):
        # 绘制波形图
        librosa.display.waveplot(ys[c], sr=srs[c], ax=ax[0, c], alpha=1)
        # 绘制频谱图
        D = librosa.amplitude_to_db(np.abs(librosa.stft(ys[c])), ref=np.max)
        librosa.display.specshow(D, y_axis="log", x_axis="time",
                                 sr=srs[c], ax=ax[1, c])
        ax[0, c].set(title=file_path[c])
        ax[0, c].label_outer()
        ax[1, c].label_outer()

    # 绘制波形图
    # librosa.display.waveplot(y, sr=sr, ax=ax[0], alpha=0.2)
    # 绘制频谱图
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # librosa.display.specshow(D, y_axis="log", x_axis="time",
    #                          sr=sr, ax=ax[1])
    print(f"file: {file_path}\tnum of info pieces: {len(annotation_info)}")
    for idx in range(len(annotation_info)):
        info = annotation_info.iloc[idx]
        t_min, t_max = info["t_min"], info["t_max"]
        f_min, f_max = info["f_min"], info["f_max"]
        start, end, audio = extract_data_by_time(y, sr, t_min, t_max)
        print(f"info[{idx}]: species_id: {info['species_id']}\tstart: ({start} / {t_min}s)\tend: ({end} / {t_max}s)\t"
              f"f_min: {f_min}Hz  f_max: {f_max}Hz\naudio data: {audio}")

        for i in range(ncols):
            nth = (0, i)

            color = "g" if info["positive"] == 1 else "r"
            ax[nth].plot(np.linspace(t_min, t_max, len(audio)), np.array(audio), color=color, alpha=.5)
            ax[nth].text(t_min, 0.1, info["species_id"], color="k")
            ax[nth].axvline(t_min, color=color)
            ax[nth].axvline(t_max, color=color)

            nth = (1, i)
            color = "g" if info["positive"] == 1 else "b"
            ax[nth].text(t_min, (f_max+f_min)/2, info["species_id"], color="white")
            ax[nth].axvline(t_min, color=color)
            ax[nth].axvline(t_max, color=color)
            ax[nth].add_patch(plt.Rectangle(xy=(t_min, f_min),width=t_max - t_min, height=f_max - f_min,
                                          color=color, fill=False, linewidth=2))
    # ax[0].set(title=file_path)
    # ax[0].label_outer()
    # ax[1].label_outer()
    print()
    plt.show()


def compare_raw_denoised():
    base_dir = "../data/denoised/"

    train_tp = pd.read_csv("../data/train_tp.csv")
    train_tp["positive"] = [1] * len(train_tp)
    train_fp = pd.read_csv("../data/train_fp.csv")
    train_fp["positive"] = [0] * len(train_fp)
    train = pd.concat([train_tp, train_fp], axis=0)
    g = train.groupby("recording_id")

    ext = ""
    # 使用train中的数据
    # recordings = train["recording_id"].unique()
    # ext = "flac"
    #
    # 使用经过RNNoise去噪后的数据
    recordings = glob.glob(base_dir + "*.wav")
    recordings = [e.split("\\")[-1].split(".")[0] for e in recordings]
    ext = "wav"

    for idx in range(len(recordings)):
        record = recordings[idx]
        sample = base_dir + record + "." + ext
        annotate_wave_spec([sample, "../data/train/" + record + ".flac"], g.get_group(recordings[idx]))


if __name__ == "__main__":
    # base_dir = "../data/train/"

    # all_flac_fs = glob.glob("../data/train/*.flac")
    samples = ['../data/train\\00ad36516.flac', '../data/train\\003b04435.flac', '../data/train\\003bec244.flac',
               '../data/train\\005f1f9a5.flac', '../data/train\\006ab765f.flac', '../data/train\\0072f0839.flac',
               ]  # all_flac_fs[:20]
    #
    #
    # print(samples)
    display_specshow(samples[0])
    #
    # duration = 60
    # display_wave([samples[0], "s0_ns.wav"], duration=duration)





