import librosa
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time

audio_file_path = "track3-estrofe.wav"
# audio_file_path = "marcha.m4a"
y, sr = librosa.load(audio_file_path)

onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, fmax=8000, n_mels=256)

# clicks = librosa.clicks(frames=onset_env, sr=sr, length=len(y))
# sd.play(y+clicks, sr)

D = np.abs(librosa.stft(y))
times = librosa.times_like(D)
plt.plot(scalex= times.any(), scaley= (1 + onset_env / onset_env.max()).any(), label='Median (custom mel)')