import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import glob


def test():
    base_dir = "../data/train/"

    y, sr = librosa.load(base_dir+"00ad36516.flac")

    chroma_orig = librosa.feature.chroma_cqt(y=y, sr=sr)

    # For display purposes, let's zoom in on a 15-second chunk from the middle of the song
    idx = tuple([slice(None), slice(*list(librosa.time_to_frames([45, 60])))])

    # And for comparison, we'll show the CQT matrix as well.
    C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=7*12*3))

    fig, ax = plt.subplots(nrows=2, sharex=True)
    img1 = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max)[idx],
                                    y_axis='cqt_note', x_axis='time', bins_per_octave=12*3,
                                    ax=ax[0])
    fig.colorbar(img1, ax=[ax[0]], format="%+2.f dB")
    ax[0].label_outer()

    img2 = librosa.display.specshow(chroma_orig[idx], y_axis='chroma', x_axis='time', ax=ax[1])
    fig.colorbar(img2, ax=[ax[1]])
    ax[1].set(ylabel='Default chroma')

    plt.show()


def rnnoise(infile, outfile):
    cmd = f"rnnoise2.exe {infile} {outfile}"
    os.system(cmd)


if __name__ == "__main__":
    base_dir = "../data/wav/"
    targ_dir = "../data/denoised/"
    fl = glob.glob(base_dir+"*.wav")

    for file in fl:
        fn = file.split("\\")[-1]
        out = targ_dir + fn
        rnnoise(file, out)
        print(f"{file} denoised to {out} finished")
