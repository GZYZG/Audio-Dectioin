import torch
from hubconf import vggish
import librosa
from librosa import display as libdisplay
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

file = "./data/00ad36516.flac"
audio, sr = sf.read(file, dtype="int16")
model = vggish()  # .hub.load("torchvggish", "VGGish")
# x = model.forward("./data/00ad36516.flac")
x = model.forward(audio, sr)
print(f"{x}")
print(x.shape)
x = np.array(x.data)
fig, ax = plt.subplots()
img = libdisplay.specshow(x, x_axis="time")
fig.colorbar(img, ax=ax)
plt.show()