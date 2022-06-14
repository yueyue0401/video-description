import librosa    

import numpy as np

audio, sample_rate = librosa.load('audio10.wav', sr=16000)

print("Sample rate: {0}Hz".format(sample_rate))
print("Audio duration: {0}s".format(len(audio) / sample_rate))

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

audio = normalize_audio(audio)

print(audio.shape)

# audio = np.sum(audio, axis=1)/2



import matplotlib.pyplot as plt
powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(audio, Fs=sample_rate)

print(powerSpectrum)

from vggish_input import wavfile_to_examples

vggish_encode = wavfile_to_examples('audio10.wav')

print(vggish_encode.shape)
