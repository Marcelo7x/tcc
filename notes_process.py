import numpy as np
import librosa
import midiutil
import re
import sounddevice as sd
import time

class NotesProcess():
    
    
    def __init__(self):
        self.minimum_note = 'A2'
        self.max_note = 'E6'
        self.voiced_acc = 0.9
        self.onset_acc = 0.8
        self.p_stay_note = 0.13
        self.p_stay_silence = 0.87
        self.frame_length = 2048
        self.window_length = 1024
        self.hop_length = 512
        self.pitch_acc = 0.99
        self.spread = 0.6
        self.transition_matrix = None
        self.prob = None
        self.onset_raw = None
        self.onset_backtrack = None
        self.onset_env = None
        self.states = None
        self.piano_format = None


    def note_validate(self, string):
        reg =  r'^[A-G][0-6]$'
        regex = re.compile(reg)
        return bool(regex.match(string))

    def set_min_note(self, m_note: str):
        if self.note_validate(m_note):
            minimum_note = m_note

    def set_max_note(self,m_note: str):
        if self.note_validate(m_note):
            max_note = m_note

    def _build_transition_matrix(self, minimum_note, max_note, p_stay_note, p_stay_silence):

        midi_min = librosa.note_to_midi(minimum_note)
        midi_max = librosa.note_to_midi(max_note)
        n_notes = midi_max - midi_min + 1
        p_ = (1 - p_stay_silence) / n_notes
        p__ = (1 - p_stay_note) / (n_notes + 1)

        # Transition matrix:
        # State 0 = silencio
        # States 1, 3, 5... = inicio (onsets)
        # States 2, 4, 6... = susteim (sustains)
        np_matrix = np.zeros((2 * n_notes + 1, 2 * n_notes + 1))

        # State 0: silencio
        np_matrix[0, 0] = p_stay_silence
        for i in range(n_notes):
            np_matrix[0, (i * 2) + 1] = p_

        # States 1, 3, 5... = onsets
        for i in range(n_notes):
            np_matrix[(i * 2) + 1, (i * 2) + 2] = 1

        # States 2, 4, 6... = sustains
        for i in range(n_notes):
            np_matrix[(i * 2) + 2, 0] = p__
            np_matrix[(i * 2) + 2, (i * 2) + 2] = p_stay_note
            for j in range(n_notes):
                np_matrix[(i * 2) + 2, (j * 2) + 1] = p__

        return np_matrix


    def _calc_probabilities(self, y, minimum_note, max_note, sr, frame_length, window_length, hop_length,
                            pitch_acc, voiced_acc, onset_acc, spread):
        fmin = librosa.note_to_hz(minimum_note)
        fmax = librosa.note_to_hz(max_note)
        midi_min = librosa.note_to_midi(minimum_note)
        midi_max = librosa.note_to_midi(max_note)
        n_notes = midi_max - midi_min + 1

        self.onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        self.onsets_raw = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=self.onset_env, hop_length=hop_length, backtrack=False,)
        S = np.abs(librosa.stft(y=y))
        rms = librosa.feature.rms(S=S)
        self.onset_backtrack = librosa.onset.onset_backtrack(self.onsets_raw, rms[0])

        # clicks = librosa.clicks(frames=self.onset_backtrack, sr=sr, length=len(y))
        # sd.play(y+clicks, sr)
        # time.sleep(20)

        # F0 and voicing
        f0, voiced_flag, voiced_prob = librosa.pyin(y= y, fmin= fmin * 0.9, fmax= fmax * 1.1, sr= sr, frame_length= frame_length, win_length= window_length, hop_length= hop_length)
        tuning = librosa.pitch_tuning(f0)
        f0_ = np.round(librosa.hz_to_midi(f0 - tuning)).astype(int)
        

        P = np.ones((n_notes * 2 + 1, len(f0)))

        for t in range(len(f0)):
            # probability of silence or onset = 1-voiced_prob
            # Probability of a note = voiced_prob * (pitch_acc) (estimated note)
            if voiced_flag[t] == False:
                P[0, t] = voiced_acc
            else:
                P[0, t] = 1 - voiced_acc

            for j in range(n_notes):
                if t in self.onset_backtrack:
                    P[(j * 2) + 1, t] = onset_acc
                else:
                    P[(j * 2) + 1, t] = 1 - onset_acc

                if j + midi_min == f0_[t]:
                    P[(j * 2) + 2, t] = pitch_acc

                elif np.abs(j + midi_min - f0_[t]) == 1:
                    P[(j * 2) + 2, t] = pitch_acc * spread

                else:
                    P[(j * 2) + 2, t] = 1 - pitch_acc

        return P


    def _convert_states_to_pianoroll(self, states, minimum_note, max_note, hop_time):
        midi_min = librosa.note_to_midi(minimum_note)
        midi_max = librosa.note_to_midi(max_note)

        states_ = np.hstack((states, np.zeros(1)))

        #states
        silence = 0
        onset = 1
        sustain = 2

        my_state = silence
        output = []

        last_onset = 0
        last_offset = 0
        last_midi = 0
        for i in range(len(states_)):
            if my_state == silence:
                if int(states_[i] % 2) != 0:
                    # achou inicio
                    last_onset = i * hop_time
                    last_midi = ((states_[i] - 1) / 2) + midi_min
                    last_note = librosa.midi_to_note(last_midi)
                    my_state = onset

            elif my_state == onset:
                if int(states_[i] % 2) == 0:
                    my_state = sustain

            elif my_state == sustain:
                if int(states_[i] % 2) != 0:
                    # achou inicio
                    # para a nota anterior
                    last_offset = i * hop_time
                    my_note = [last_onset, last_offset, last_midi, last_note]
                    output.append(my_note)

                    # comeca a nova nota
                    last_onset = i * hop_time
                    last_midi = ((states_[i] - 1) / 2) + midi_min
                    last_note = librosa.midi_to_note(last_midi)
                    my_state = onset

                elif states_[i] == 0:
                    # achou silencio.
                    # para a nota anterior.
                    last_offset = i * hop_time
                    my_note = [last_onset, last_offset, last_midi, last_note]
                    output.append(my_note)
                    my_state = silence

        return output

    def _convert_pianoroll_to_midi(self, y, sr, pianoroll):
        # onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bpm = librosa.feature.tempo(y=y, onset_envelope=self.onset_env, sr=sr)

        quarter_note = 60 / bpm
        ticks_per_quarter = 1024

        onsets = np.array([p[0] for p in pianoroll])
        offsets = np.array([p[1] for p in pianoroll])

        onsets = onsets / quarter_note
        offsets = offsets / quarter_note
        durations = offsets - onsets

        MyMIDI = midiutil.MIDIFile(1)
        MyMIDI.addTempo(0, 0, bpm)

        for i in range(len(onsets)):
            MyMIDI.addNote(0, 0, int(pianoroll[i][2]), onsets[i], durations[i], 100)

        return MyMIDI
    
    

    def process(self, y, sr):
        transition_matrix = self._build_transition_matrix(self.minimum_note, self.max_note, self.p_stay_note, self.p_stay_silence)
        prob = self._calc_probabilities(y, self.minimum_note, self.max_note, sr, self.frame_length, self.window_length,
                                    self.hop_length, self.pitch_acc, self.voiced_acc, self.onset_acc, self.spread)
        init_mat = np.zeros( transition_matrix.shape[0])
        init_mat[0] = 1
        self.states = librosa.sequence.viterbi(prob, transition_matrix, p_init=init_mat)

    def highpass_filter(self, y, sr):
        filter_stop_freq = 70  # Hz
        filter_pass_freq = 100  # Hz
        filter_order = 1001

        # High-pass filter
        nyquist_rate = sr / 2.
        desired = (0, 0, 1, 1)
        bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
        filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

        # Apply high-pass filter
        filtered_audio = signal.filtfilt(filter_coefs, [1], y)
        return filtered_audio

    def getNotesPianoFormart(self, y, sr):
        audio_file_path = "track3-estrofe.wav"
        # audio_file_path = "marcha.m4a"
        y, sr = librosa.load(audio_file_path)
        # y = self.highpass_filter(y, sr)
        # y = librosa.util.normalize(y)
        self.process(y, sr)
        piano_format = self._convert_states_to_pianoroll(self.states, self.minimum_note, self.max_note, self.hop_length/sr)
        # print(piano_format)
        # print(f'len states {len(self.states)}')
        print(f'len piano {len([p[0] for p in piano_format])}')
        print(f'{([p[0] for p in piano_format])}')
        # print(f'len onsets {len(self.onset_backtrack)}')
        print(f'{len(librosa.frames_to_time(self.onset_backtrack, sr=sr))}')
        print(f'times onsets {librosa.frames_to_time(self.onset_backtrack, sr=sr)}')

        cont = 0
        for el in piano_format:
            for ell in librosa.frames_to_time(self.onset_backtrack, sr=sr):
                if float("{:.2f}".format(el[0])) == float("{:.2f}".format(ell)):
                    cont+=1
                    break  
        print("Contttttttt  " + str(cont))
        # return piano_format
        self.toMidi(y=y, sr=sr, piano_format=piano_format)

    def toMidi(self, y, sr, piano_format):
        midi_format = self._convert_pianoroll_to_midi(y, sr, piano_format)
        with open("out.mid", "wb") as output_file:
            midi_format.writeFile(output_file)
    


# NotesProcess().getNotesPianoFormart(y=None, sr=None)
if __name__ == '__main__':
    notes_p = NotesProcess()
    notes_p.getNotesPianoFormart(y=None, sr=None)