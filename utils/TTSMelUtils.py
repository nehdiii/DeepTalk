import numpy as np
import librosa
import os, copy
from scipy import signal
import torch as th
REF_DB = 20
MAX_DB = 100
POWER  = 1.2 # Exponent for amplifying the predicted magnitude
PREEMPHASIS = 0.97
N_ITER = 60
SR = 22050
FRAME_SHIFT = 0.0125 # seconds
FRAME_LENGTH= 0.05 # seconds
N_FFT =  2048
HOP_LENGTH = int(SR*FRAME_SHIFT)  # samples.
WIN_LENGTH = int(SR*FRAME_LENGTH)  # samples.


def spectrogram2wav(mag):
  """# geneartes wave file from linear magnitude spectrograme
  args :
  mag : a numpy array of (T,1+n_fft//2)
  returns : a 1-D numpy array
  """
  # transpose
  mag = mag.T

  # de-normlize
  mag = (np.clip(mag,0,1)*MAX_DB)-MAX_DB+REF_DB

  # to_amplitude
  mag = np.power(10.0,mag*0.05)

  # wav recontruction
  wav = f=griffin_lim(mag**POWER)

  # de-preemphasis
  wav = signal.lfilter([1],[1,-PREEMPHASIS],wav)

  # trim
  wav,_ = librosa.effects.trim(wav,top_db=10)

  return wav.astype(np.float32)


def griffin_lim(spectrogram):
  # spectrogram ---- > wav
  X_best = copy.deepcopy(spectrogram)
  for i in range(N_ITER):
    X_t = invert_spectrogram(X_best)
    est = librosa.stft(X_t,N_FFT,HOP_LENGTH,WIN_LENGTH)
    phase = est / np.maximum(1e-8,np.abs(est))
    X_best = spectrogram*phase
  X_t = invert_spectrogram(X_best)
  y = np.real(X_t)
  return y


def invert_spectrogram(spectrogram):
    """
    Applies inverse fft
    args : spectrogram : [1+n_fft/2,t]

    """
    return librosa.istft(spectrogram,
                         HOP_LENGTH,
                         WIN_LENGTH,
                         window="hann")
