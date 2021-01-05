import librosa
import librosa.display
import pandas as pd
import soundfile as sf
import glob, os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def annimation_test():
    def update_points(num):
        '''
        更新数据点
        '''
        point_ani.set_data(x[num], y[num])
        return point_ani,
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(x, y)
    ax.grid(ls="--")
    point_ani, = ax.plot(x[0], y[0], "ro")
    ani = animation.FuncAnimation(fig, update_points, np.arange(0, 100), interval=60, blit=True)
    plt.show()
    ani.save('sin_test2.gif', writer='imagemagick', fps=10)
    pass


def wave_animation(file_path):
    def update(idx):
        print(y[idx])
        vline.set_data([idx/sr]*2, [-1, 1])
        # vline.set_linewidth(np.random.standard_normal())
        return vline,
    y, sr = librosa.load(file_path)
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    librosa.display.waveplot(y, sr=sr, ax=ax, alpha=1)
    ax.grid(ls="--")
    vline = ax.plot([0,0], [-abs(y[0]), abs(y[0])], color="r", lw=.5)[0]
    interval = 100
    ani = animation.FuncAnimation(fig, update, np.arange(0, len(y), int(sr*interval/1000)), interval=100, blit=True)
    # ani.save('sin_test2.gif', writer='pillow', fps=10)
    plt.show()


def bar_animation():
    def animate(frame):

        for bar in bars:
            bar.set_height(np.random.standard_normal())
        return bars,

    fig, ax = plt.subplots()
    x = list(range(0,25))
    y = np.random.standard_normal(len(x))
    bars = ax.bar(x, y, .5)

    ani = animation.FuncAnimation(fig, animate, 100, interval=100, blit=False, repeat=True)

    plt.show()


def split(right=60):
    left = 0
    # right = 60
    frame_length = 0.025
    piece_length = frame_length * 8
    stride = 0.1
    ends = []
    while True:
            s = left
            end = int(sr * (s+piece_length))
            left += stride
            ends.append(end)
            if s+piece_length > right:
                break
    return ends


def animate_real_time_pred(wave_data, sr, preds, labels, save_pth="realtime.gif"):
    def animate(frame):
        vline.set_data([ends[frame] / sr] * 2, [-2, 2])
        # appear = (preds[frame] == 1)
        for idx, bar in enumerate(bars):
            bar.set_height(preds[frame][idx])
            if preds[frame][idx] >= 0:
                bar.set_color('g')
            else:
                bar.set_color('r')

        for idx, bar in enumerate(t_bars):
            bar.set_height(labels[frame][idx])
            if labels[frame][idx] >= 0:
                bar.set_color('g')
            else:
                bar.set_color('r')

        # for e in appear:
        #     if e != 24:
        #         ax[1].text(e, 0.5, f"{e}", color="k", size="large")

        return bars,vline,t_bars,
    # print(preds)
    fig, ax = plt.subplots(ncols=1, nrows=3)
    librosa.display.waveplot(wave_data, sr=sr, ax=ax[0], alpha=1)
    vline = ax[0].plot([0, 0], [-2, 2], color="r", lw=1.5)[0]
    x = list(range(25))
    y_init = [0] * len(x)
    y_init[-1] = 1

    bars = ax[1].bar(x, y_init, .5, align="center")
    ax[1].set_xticks(x)
    ax[1].set_ylim(-1,1)

    t_bars = ax[2].bar(x, y_init, .5, align="center")
    ax[2].set_xticks(x)
    ax[2].set_ylim(-1, 1)

    interval = 100
    ends = split(right=len(wave_data)/sr)
    ani = animation.FuncAnimation(fig, animate, frames=len(ends), interval=interval, blit=False, repeat=True)
    ani.save(save_pth, writer='pillow', fps=10)
    plt.show()

    pass


if __name__ == "__main__":
    # 使用matplotlib展示动画
    # annimation_test()
    # 以动画的形式展示声波 400b7210c 5f8eecc9e
    file_path = "../dataproc/species.flac"
    wave_data, sr = librosa.load(file_path)
    data_path = "../solution3/species_preds_labelsmodel08.npz"
    data = np.load(data_path, allow_pickle=True)
    animate_real_time_pred(wave_data, sr, data['preds'], data["labels"], save_pth=f'realtime_{data_path.split("/")[-1].split(".")[0]}.gif')
    # wave_animation(file_path)
    # bar_animation()

    # base_dir = "../data/train/

    # train = pd.read_csv("../data/train_all.csv")
    # gs = train.groupby("recording_id")
    #
    # all_flac_fs = os.listdir("../data/train/")
    # samples = ['../data/train\\00ad36516.flac', '../data/train\\003b04435.flac', '../data/train\\003bec244.flac',
    #            '../data/train\\005f1f9a5.flac', '../data/train\\006ab765f.flac', '../data/train\\0072f0839.flac',
    #            ]  # all_flac_fs[:20]
    # file = all_flac_fs[7]
    # record = file.split(".")[0]
    # annotate_spec(os.path.join("../data/train/", file), gs.get_group(record))
    #
    #
    # print(samples)
    # display_specshow(samples[0])

    # duration = 60
    # display_wave([samples[3]], duration=duration)





