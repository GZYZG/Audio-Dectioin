import torch
from hubconf import vggish
import librosa
from librosa import display as libdisplay
import matplotlib.pyplot as plt
import numpy as np



model = vggish()#torch.hub.load("torchvggish", "VGGish")
x = model.forward("./data/00ad36516.flac")
print(f"{x}")
x = np.array(x.data)
fig, ax = plt.subplots()
img = libdisplay.specshow(x, x_axis="time")
fig.colorbar(img, ax=ax)
plt.show()