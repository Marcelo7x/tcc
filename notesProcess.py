import numpy as np
import librosa
import midiutil
import re
import scipy.signal as signal

class NotesProcess():
    
    
    def __init__(self):
        self.minimum_note = 'A2'
        self.max_note = 'E6'
        self.voiced_acc = 0.9
        self.onset_acc = 0.8
        self.frame_length = 2048
        self.window_length = 1024
        self.hop_length = 256
        self.pitch_acc = 0.99
        self.spread = 0.6
        self.transition_matrix = None
        self.prob = None
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


    """
    Retorna a matriz de transição com um estado de silêncio e dois estados
    (início e sustentação) para cada nota.

    Parâmetros:
        minimum_note : string, formato 'A#4' (Nota mais baixa suportada por esta matriz de transição.)
        max_note : string, formato 'A#4'(Nota mais alta suportada por esta matriz de transição.)
        p_stay_note : float, entre 0 e 1 (Probabilidade de um estado de sustentação retornar a si mesmo.)
        p_stay_silence : float, entre 0 e 1 (Probabilidade do estado de silêncio retornar a si mesmo.)

    Retorno: Uma matriz de transição 2x2 em que T[i,j] é a probabilidade de ir do estado i para o estado j.


    A função build_transition_matrix converte as notas mínima e máxima para sua representação MIDI utilizando a função librosa.note_to_midi.
    Em seguida, calcula o número de notas abrangidas pela matriz de transição, n_notes,
    que é obtido pela diferença entre os valores MIDI das notas máxima e mínima mais 1.

    A função então calcula as probabilidades p_ e p__. p_ é a probabilidade de transição
    para os estados de início (onsets), exceto o estado de silêncio, e é calculada dividindo
    a probabilidade restante (1 - p_stay_silence) pelo número de notas mais 1. p__ é a probabilidade
    de transição para os estados de sustentação (sustains), incluindo o estado de silêncio, e é calculada
    dividindo a probabilidade restante (1 - p_stay_note) pelo número de notas mais 1.

    A matriz de transição np_matrix é inicializada como uma matriz 2x2 com zeros, onde o número
    de linhas e colunas é igual a 2 * n_notes + 1. Os estados na matriz são mapeados da seguinte forma:

    Estado 0: silêncio
    Estados 1, 3, 5...: início (onsets)
    Estados 2, 4, 6...: sustentação (sustains)
    A seguir, o código preenche a matriz de transição com as probabilidades adequadas. Aqui está o que cada parte do código faz:

    Estado 0 (silêncio): A probabilidade de transição para o próprio estado é definida como p_stay_silence.
    A probabilidade de transição para os estados de início é definida como p_. O loop for percorre as notas
    e define as probabilidades de transição correspondentes na matriz.
    Estados 1, 3, 5... (onsets): A probabilidade de transição para o próximo estado é definida como 1.
    Isso significa que os estados de início sempre transicionam para os estados de sustentação correspondentes.
    Estados 2, 4, 6... (sustains): A probabilidade de transição de um estado de sustentação para o estado de silêncio
    é definida como p__. A probabilidade de transição para o próprio estado de sustentação é definida como p_stay_note.
    O loop for interno percorre as notas e define as probabilidades de transição correspondentes para os estados de início
    """

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


    """"
        Estimar probabilidades anteriores a partir do sinal de áudio

        Parâmetros
        ----------
        y: array numpy 1-D contendo amostras de áudio
        minimum_note: string, formato 'A#4'
            Nota mais baixa suportada por este estimador
        max_note: string, formato 'A#4'
            Nota mais alta suportada por este estimador
        sr: int
            Taxa de amostragem.
        frame_length: int
        window_length: int
        hop_length: int
            Parâmetros para a estimativa FFT
        pitch_acc: float, entre 0 e 1
            Probabilidade (estimada) de que o estimador de pitch esteja correto.
        voiced_acc: float, entre 0 e 1
            Precisão estimada do parâmetro "voiced".
        onset_acc: float, entre 0 e 1
            Precisão estimada do detector de início.
        spread: float, entre 0 e 1
            Probabilidade de que o cantor/músico tenha uma desvio de um semitom
            devido a vibrato ou glissando.
        Retorna
        -------
        Matriz 2D em que P[j,t] é a probabilidade anterior de estar no estado j no tempo


    calc_probabilities primeiro converte as notas mínima e máxima para sua frequência em Hz
    e suas representações MIDI utilizando as funções librosa.note_to_hz e librosa.note_to_midi, 
    respectivamente. Em seguida, calcula o número de notas abrangidas pelo estimador, n_notes, 
    que é obtido pela diferença entre os valores MIDI das notas máxima e mínima mais 1.

    A função prossegue estimando o pitch (frequência fundamental) e o parâmetro "voiced" (voz ativa) 
    do sinal de áudio utilizando a função librosa.pyin. A função também calcula o ajuste de afinação 
    (tuning) e arredonda as frequências fundamentais estimadas para as representações MIDI mais próximas em f0_.

    Em seguida, a função utiliza o detector de início librosa.onset.onset_detect para identificar 
    os momentos de início no sinal de áudio.

    A matriz de probabilidades P é inicializada como uma matriz 2D de dimensões (n_notes * 2 + 1, len(f0)), 
    onde n_notes * 2 + 1 representa o número de estados possíveis (silêncio, onsets e sustains) e len(f0) 
    é o número de momentos de tempo.

    A seguir, o código preenche a matriz de probabilidades com as probabilidades adequadas. 
    Aqui está o que cada parte do código faz:

    Para cada momento de tempo t, o código calcula a probabilidade de estar no estado de silêncio ou de início. 
    Se o sinal de áudio for considerado não vocalizado (voiced_flag[t] == False), a probabilidade de estar no estado 
    de silêncio é definida como voiced_acc. Caso contrário, a probabilidade de estar no estado de silêncio é definida como 1 - voiced_acc.
    Para cada estado de início (onset) j, o código verifica se o momento de tempo t está entre os momentos de início 
    detectados. Se estiver, a probabilidade de estar no estado de início é definida como onset_acc. Caso contrário, 
    a probabilidade de estar no estado de início é definida como 1 - onset_acc.
    Para cada estado de sustentação (sustain) j, o código verifica se o valor arredondado da frequência fundamental 
    estimada f0_[t] corresponde à nota MIDI j + midi_min. Se corresponder, a probabilidade de estar no estado de 
    sustentação é definida como pitch_acc. Caso contrário, o código verifica se a diferença absoluta entre a nota 
    estimada e a nota MIDI é igual a 1. Se for, a probabilidade de estar no estado de sustentação é definida como 
    pitch_acc * spread. Caso contrário, a probabilidade de estar no estado de sustentação é definida como 1 - pitch_acc.


    Uma probabilidade posterior é a probabilidade de atribuir observações a grupos considerando-se os dados. 
    Uma probabilidade anterior é a probabilidade de que uma observações estará em um grupo antes de se coletar os dados. 
    Por exemplo, se você está classificando os compradores de um carro específico, você poderia saber de antemão que 60% dos 
    compradores são homens e 40% são mulheres. Se você conhece ou pode estimar essas probabilidades, uma análise 
    discriminante pode usar essas probabilidades anteriores para calcular as probabilidades posteriores.


    "Estimar probabilidades anteriores" se refere ao processo de calcular a probabilidade de estar em determinados 
    estados musicais em momentos anteriores com base em informações disponíveis. No contexto desse código, as probabilidades
    anteriores são usadas para modelar a probabilidade de estar em estados específicos (silêncio, início, sustentação) 
    em cada momento de tempo, com base em características do sinal de áudio.

    Através da análise do sinal de áudio, como a estimativa do pitch, a detecção de início e outras características, 
    é possível fazer suposições probabilísticas sobre a presença de certos estados musicais em cada momento de tempo. 
    Essas suposições são usadas para construir uma matriz de probabilidades anteriores, em que cada elemento representa 
    a probabilidade de estar em um estado específico em um determinado momento de tempo.

    Ao estimar probabilidades anteriores, o código considera informações como a presença de voz, a presença de inícios 
    musicais e a correspondência entre o pitch estimado e as notas musicais. Essas informações são usadas para atribuir 
    probabilidades aos estados de silêncio, início e sustentação em cada momento de tempo.

    Essas probabilidades anteriores podem ser usadas como entrada para modelos de sequência musical, como cadeias de Markov,
    onde as probabilidades de transição entre os estados são influenciadas por essas probabilidades anteriores. 
    Dessa forma, é possível modelar a evolução temporal da música com base nas características do sinal de áudio.
    """

    def _calc_probabilities(self, y, minimum_note, max_note, sr, frame_length, window_length, hop_length,
                            pitch_acc, voiced_acc, onset_acc, spread):
        fmin = librosa.note_to_hz(minimum_note)
        fmax = librosa.note_to_hz(max_note)
        midi_min = librosa.note_to_midi(minimum_note)
        midi_max = librosa.note_to_midi(max_note)
        n_notes = midi_max - midi_min + 1

        # F0 and voicing
        f0, voiced_flag, voiced_prob = librosa.pyin(y= y, fmin= fmin * 0.9, fmax= fmax * 1.1, sr= sr, frame_length= frame_length, win_length= window_length, hop_length= hop_length)
        tuning = librosa.pitch_tuning(f0)
        f0_ = np.round(librosa.hz_to_midi(f0 - tuning)).astype(int)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)

        P = np.ones((n_notes * 2 + 1, len(f0)))

        for t in range(len(f0)):
            # probability of silence or onset = 1-voiced_prob
            # Probability of a note = voiced_prob * (pitch_acc) (estimated note)
            # Probability of a note = voiced_prob * (1-pitch_acc) (estimated note)
            if voiced_flag[t] == False:
                P[0, t] = voiced_acc
            else:
                P[0, t] = 1 - voiced_acc

            for j in range(n_notes):
                if t in onsets:
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


    """
        Converte a sequência de estados para uma notação intermediária interna em formato de piano-roll

        Parâmetros:
        states : int (Sequência de estados estimados)
        minimum_note : string, formato 'A#4' ()
        max_note : string, formato 'A#4'
        hop_time : float (Intervalo de tempo entre dois estados)

        Retorna:
        output : Lista de listas
        output[i] é a i-ésima nota na sequência. Cada nota é uma lista descrita por [tempo_inicial, 	tempo_final, altura].
    """

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
                    #achou inicio
                    last_onset = i * hop_time
                    last_midi = ((states_[i] - 1) / 2) + midi_min
                    last_note = librosa.midi_to_note(last_midi)
                    my_state = onset


            elif my_state == onset:
                if int(states_[i] % 2) == 0:
                    my_state = sustain

            elif my_state == sustain:
                if int(states_[i] % 2) != 0:
                    #achou inicio
                    #para a nota anterior
                    last_offset = i * hop_time
                    my_note = [last_onset, last_offset, last_midi, last_note]
                    output.append(my_note)

                    #comeca a nova nota
                    last_onset = i * hop_time
                    last_midi = ((states_[i] - 1) / 2) + midi_min
                    last_note = librosa.midi_to_note(last_midi)
                    my_state = onset

                elif states_[i] == 0:
                    #achou silencio. 
                    # para a nota anterior.
                    last_offset = i * hop_time
                    my_note = [last_onset, last_offset, last_midi, last_note]
                    output.append(my_note)
                    my_state = silence

        return output


    def _convert_pianoroll_to_midi(self, y, sr, pianoroll):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bpm = librosa.feature.tempo(y=y, onset_envelope=onset_env, sr=sr)

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
        transition_matrix = self._build_transition_matrix(self.minimum_note, self.max_note, 0.15, 0.7)
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
        # audio_file_path = "pastor.m4a"
        # y, sr = librosa.load(audio_file_path)
        y = self.highpass_filter(y, sr)
        self.process(y, sr)
        piano_format = self._convert_states_to_pianoroll(self.states, self.minimum_note, self.max_note, self.hop_length / sr)
        # print(piano_format)
        return piano_format
        # self.toMidi(y=y, sr=sr, piano_format=piano_format)

    def toMidi(self, y, sr, piano_format):
        midi_format = self._convert_pianoroll_to_midi(y, sr, piano_format)
        with open("out.mid", "wb") as output_file:
            midi_format.writeFile(output_file)


# NotesProcess().getNotesPianoFormart(y=None, sr=None)
if __name__ == '__main__':
    notes_p = NotesProcess()
    notes_p.getNotesPianoFormart(y=None, sr=None)