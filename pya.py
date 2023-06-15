import pyaudio
import numpy as np
import wave
import librosa as lbr
import argparse

n_fft = 256
channels = 1
sr = 44100
n = 2

chunk = 2*n_fft

pyaudio.get_portaudio_version()

pa = pyaudio.PyAudio()

default_in = pa.get_default_host_api_info()['defaultInputDevice']
default_out = pa.get_default_host_api_info()['defaultOutputDevice']

stream_in = pa.open(
    rate=sr,
    channels=channels,
    format=pyaudio.paInt16,
    input=True,
    input_device_index=3,         # input device index
    frames_per_buffer=chunk
)

stream_out = pa.open(
    rate=sr,
    channels=channels,
    format=pyaudio.paInt16,
    output=True,
    output_device_index=1,         # input device index
    frames_per_buffer=chunk
)


# output_filename = 'audio-recording.wav'
# wav_file = wave.open(output_filename, 'wb')

# define audio stream properties
# wav_file.setnchannels(channels)        # number of channels
# wav_file.setsampwidth(2)        # sample width in bytes
# wav_file.setframerate(sr)    # sampling rate in Hz
data = []
try:
    print("gravando")

    while True:
        input_audio = stream_in.read(chunk)
        y = np.frombuffer(input_audio, dtype=np.int16).astype(float)
        data.append(y)
#   audio_shift = lbr.effects.pitch_shift(y, sr=sr, n_steps=n, n_fft=n_fft)
#   audio_shift = audio_shift.astype(np.int16).tobytes()
#   stream_out.write(audio_shift)
#   # write samples to the file
#   wav_file.writeframes(audio_shift)
except KeyboardInterrupt:
    print("parando")


print(data)

# import way_to_midi as wtm
# wtm.main(buffer=y)