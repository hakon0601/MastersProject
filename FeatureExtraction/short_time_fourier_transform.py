import sys
from math import floor

from FeatureExtraction.feature_extractor_base import FeatureExtractorBase
import librosa
import numpy as np


class STFT(FeatureExtractorBase):
    def __init__(self, fft_window_size=2048):
        self.fft_window_size = fft_window_size
        self.output_size = 0

    def extract_features(self, samples):
        processed_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            processed_sample = librosa.stft(sample, self.fft_window_size)
            magnitude = np.abs(processed_sample) # the magnitude of frequency bin f at frame t
            phase = np.angle(processed_sample) # the phase of frequency bin f at frame t
            concatenation = magnitude.flatten() + phase.flatten()

            processed_samples.append(concatenation)
            sys.stdout.write("\rExtracting features %d%%" % floor((i + 1) * (100/len(samples))))
            sys.stdout.flush()
        print()
        self.output_size = len(processed_samples[0])
        return processed_samples