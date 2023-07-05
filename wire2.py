import sounddevice as sd
import numpy as np
import notesProcess as pn
import time
import os
import threading

class AudioProcessor:
    def __init__(self):
        self.figureOfTime = 1
        self.beats = 4
        self.going = 60
        self.input_device = None
        self.output_device = None
        self.channels = 2
        self.dtype = None
        self.samplerate = None
        self.blocksize = None
        self.latency = None
        self.data = []
        self.stop = False

        self.load_config()

        self.beatTime = 60.0 / self.going
        self.compassTime = 4 * self.figureOfTime * self.beatTime
        self.windowSize = self.blocksize / self.samplerate
        self.windowPerBeat = self.beatTime / self.windowSize
        self.windowPerCompasse = self.beats * self.windowPerBeat

    def load_config(self):
        with open('config.env', 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                setattr(self, key.strip(), int(value.strip()))

    def process_audio(self):
        print("Processando áudio")
        
        while not self.stop:
            if len(self.data) >= self.windowPerCompasse:
                rec = np.array((len(self.data) * self.samplerate,))
                rec = np.concatenate(np.squeeze(np.stack(self.data)))
                del self.data[:]
                print(pn.NotesProcess().getNotesPianoFormart(y=rec, sr=self.samplerate))

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.data.append(np.frombuffer(indata, dtype=np.int16).astype(float))

    def capture_audio(self):
        try:
            with sd.InputStream(device=self.input_device,
                               samplerate=self.samplerate,
                               blocksize=self.blocksize,
                               dtype="int16",
                               latency=self.latency,
                               channels=self.channels,
                               callback=self.callback):
                print('#' * 80)
                print('Press Return to quit')
                print('#' * 80)
                input()
                # time.sleep(2 * self.beats * self.figureOfTime * self.beatTime)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(type(e).__name__ + ': ' + str(e))
            exit()

        print('stop')

    def start_processing(self):
        thread_audio = threading.Thread(target=self.capture_audio)
        thread_audio.start()

        thread_process = threading.Thread(target=self.process_audio)
        thread_process.start()

        thread_audio.join()
        self.stop = True
        thread_process.join()

if __name__ == '__main__':
    audio_processor = AudioProcessor()
    audio_processor.start_processing()
