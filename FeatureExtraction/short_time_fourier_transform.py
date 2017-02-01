import sys
from math import floor

from FeatureExtraction.feature_extractor_base import FeatureExtractorBase
import librosa
import numpy as np
import matplotlib.pyplot as plt
from run_config_settings import *


class STFT(FeatureExtractorBase):
    def __init__(self, fft_window_size=2048):
        self.fft_window_size = fft_window_size

    def extract_features(self, samples):
        processed_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            processed_sample = librosa.stft(sample, self.fft_window_size) # Creates an array of touples, complex values? fix #TODO time-freq?
            a = np.abs(processed_sample) # the magnitude of frequency bin f at frame t
            b = np.angle(processed_sample) # the phase of frequency bin f at frame t
            c = a.flatten() + b.flatten() # #TODO find a good way to do this

            processed_samples.append(c)
            sys.stdout.write("\rExtracting features %d%%" % floor((i + 1) * (100/len(samples))))
            sys.stdout.flush()
        print()
        return processed_samples