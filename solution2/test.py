import soundfile as sf
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import resampy

file = "../data/train/1100df69a.flac"
samplerate = 48000
(sig, rate) = sf.read(file, dtype="int16")
if rate != samplerate:
    sig = resampy.resample(sig, rate, samplerate)
mfcc_feat = mfcc(sig, rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbanks = logfbank(sig, rate)


