import sys
from math import floor
from FeatureExtraction.feature_extractor_base import FeatureExtractorBase
import librosa
import numpy as np


class MFCC(FeatureExtractorBase):
    def __init__(self, n_mfcc=20):
        self.n_mfcc = n_mfcc
        self.output_size = 0

    def extract_features(self, samples):
        processed_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            processed_sample = librosa.feature.mfcc(sample, n_mfcc=self.n_mfcc)
            magnitude = np.abs(processed_sample)   # the magnitude of frequency bin f at frame t
            phase = np.angle(processed_sample) # the phase of frequency bin f at frame t
            concatenation = magnitude.flatten() + phase.flatten()

            processed_samples.append(concatenation)
            sys.stdout.write("\rExtracting features %d%%" % floor((i + 1) * (100/len(samples))))
            sys.stdout.flush()
        print()
        self.output_size = len(processed_samples[0])
        return processed_samples