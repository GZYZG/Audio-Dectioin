#!/usr/bin/env python
# coding: utf-8

# # ASR中常用的语音特征之FBank和MFCC（原理 + Python实现）

# 一步一步讲解和实现ASR中常用的语音特征——FBank和MFCC的提取，包括算法原理、代码和可视化等。
# 
# 完整Jupyter Notebook链接：https://github.com/Magic-Bubble/SpeechProcessForMachineLearning/blob/master/speech_process.ipynb

# ## 语音信号的产生

# 语音通常是指人说话的声音。从生物学的角度来看，是气流通过声带、咽喉、口腔、鼻腔等发出声音；从信号的角度来看，不同位置的震动频率不一样，最后的信号是由基频和一些谐波构成。
# 
# <img src='./resources/speech_production_model.png' width='55%'>
# 
# 
# 之后被设备接收后（比如麦克风），会通过A/D转换，将模拟信号转换为数字信号，一般会有采样、量化和编码三个步骤，采样率要遵循奈奎斯特采样定律：$fs >= 2f$，比如电话语音的频率一般在300Hz~3400Hz，所以采用8kHz的采样率足矣。
# 
# 下面采用一个30s左右的16比特PCM编码后的语音wav为例。

# ## 准备工作

# ### 1. 导包

# In[8]:


import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
# get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2. 绘图工具

# In[2]:


# 绘制时域图
def plot_time(signal, sample_rate):
    time = np.arange(0, len(signal)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, signal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()


# In[22]:


# 绘制频域图
def plot_freq(signal, sample_rate, fft_size=512):
    xf = np.fft.rfft(signal, fft_size) / fft_size
    freqs = np.linspace(0, sample_rate/2, int(fft_size/2 + 1))
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.figure(figsize=(20, 5))
    plt.plot(freqs, xfp)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()


# In[23]:


# 绘制频谱图
def plot_spectrogram(spec, note):
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()

def filter_banks(file, normalization=True):
    signal, sample_rate = sf.read(file, dtype='int16')
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frame_size, frame_stride = 0.025, 0.01
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1,
                                                                                                                    1)
    frames = pad_signal[indices]
    hamming = np.hamming(frame_length)
    frames *= hamming
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    nfilt = 40
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    for i in range(1, nfilt + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    if normalization:
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

    return filter_banks

    # num_ceps = 12
    # mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)]
    #

def mfcc_efficient(file, normalization=True):
    fbanks = filter_banks(file, normalization=False)
    num_ceps = 12
    mfcc = dct(fbanks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)]

    if normalization:
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc


if __name__ == "__main__":
    file = "../data/train/1100df69a.flac"

    (sig, rate) = sf.read(file, dtype="int16")
    mfcc_feat = mfcc(sig, rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbanks = logfbank(sig, rate)
    plot_spectrogram(fbanks.T, 'Filter Banks')
    fbanks = filter_banks(file)
    plot_spectrogram(fbanks.T, 'Filter Banks')

    dpi = 100
    fig = plt.figure()  # plt.figure(num=None, figsize=(300 / dpi, 300 / dpi), dpi=dpi)
    plt.subplot(222)
    ax = plt.axes()
    ax.set_axis_off()
    librosa.display.specshow(fbanks.T)
    plt.savefig(file.split("/")[-1].split(".")[0]+".png", bbox_inches='tight',
                   transparent=True, pad_inches=0.0)

    plt.show()