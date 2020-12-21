import soundfile as sf
import pandas as pd
import numpy as np
import librosa
import glob

"""
for f in fl:
     fn = f.split("\\")
     fn = fn[-1]
     fn = fn.split(".")[0]+".wav"
     os.system("rnnoise-windows.exe {} ./denoise/{}".format(f, fn))
"""

def flac_to_wav(infile, outfile):
    data, sr = sf.read(infile)
    sf.write(outfile, data, samplerate=sr, subtype="PCM_16")


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


def concat_data_to_audio(data_list, sr, outfile):
    data = []
    for e in data_list:
        data.extend(e)
    sf.write(outfile, data, sr, format="PCM_16")


def sample_ranges(ranges, range_len=.8, max_range_len=1):
    """
    从多个区间里随机采样等距的区间，注意，采样到的区间可能会重叠
    :param ranges: 被采样的多个区间
    :param range_len: 采样的区间长度
    :param max_range_len: 每个被采样区间的最小长度
    :return:
    """
    sampled_ranges = []
    for r in ranges:
        l = r[1] - r[0]
        sample_points = []
        sample_num = int(l // max_range_len)
        for p in range(sample_num):
            ep = np.random.rand()
            sample_points.append(r[0] + l * ep)
            sampled_ranges.append((sample_points[-1], sample_points[-1] + range_len))

    return sampled_ranges


if __name__ == "__main__":
    base_dir = "../data/train/"
    train_tp = pd.read_csv("../data/train_tp.csv")
    g = train_tp.groupby("recording_id")
    recordings = train_tp["recording_id"].unique().tolist()

    noise_data = []
    non_noise_data = []
    for i in range(50):
        ranges = []
        record = recordings[i]
        group = g.get_group(record)
        for ii in range(len(group)):
            item = group.iloc[ii]
            ranges.append((item["t_min"], item["t_max"]))
        ranges.sort(key=lambda x: x[0])

        noise_ranges = []
        start = 0
        for r in ranges:
            end = r[0] - 0.001
            if abs(end - start) < 1e-05:
                continue
            noise_ranges.append((start, end))
            start = r[1]+0.001
        if abs(ranges[-1][1] - 60) > .1:
            noise_ranges.append((start, 60))

        y, sr = librosa.load(base_dir+record+".flac")
        for e in ranges:
            *_, tmp = extract_data_by_time(y, sr, e[0], e[1])
            non_noise_data.extend(tmp)

        sampled_noise_ranges = []
        # 以1秒为采样个数的衡量标准
        sampled_noise_ranges = sample_ranges(noise_ranges, range_len=1.5, max_range_len=3)
        # 提取噪声区间的数据进行拼接
        for e in sampled_noise_ranges:
            *_, tmp = extract_data_by_time(y, sr, e[0], e[1])
            noise_data.extend(tmp)
        # for r in noise_ranges:
        #     l = r[1] - r[0]
        #     sample_points = []
        #     sample_num = int(l // 1)
        #     for p in range(sample_num):
        #         ep = np.random.rand()
        #         sample_points.append(r[0]+l*ep)
        #         sampled_noise_ranges.append((sample_points[-1], sample_points[-1]+0.8))
        #
        #         *_, tmp = extract_data_by_time(y, sr, sampled_noise_ranges[-1][0], sampled_noise_ranges[-1][1])
        #         noise_data.extend(tmp)

        print(f"ranges: {ranges}\tnoise_ranges: {noise_ranges}")
        print(f"sampled_noise_ranges: {sampled_noise_ranges}")

    sf.write("./non_noise.wav", non_noise_data, sr, subtype="PCM_16")
    sf.write("./noise.wav", noise_data, sr, subtype="PCM_16")
