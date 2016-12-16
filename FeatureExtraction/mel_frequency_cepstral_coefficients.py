import sys
from math import floor

from FeatureExtraction.feature_extractor_base import FeatureExtractorBase
import librosa
import numpy as np


class MFCC(FeatureExtractorBase):
    def __init__(self, n_mfcc=20):
        self.n_mfcc = n_mfcc

    def extract_features(self, samples, show_spectrogram=False):
        processed_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            processed_sample = librosa.feature.mfcc(sample, n_mfcc=self.n_mfcc)
            # print("mfcc", processed_sample)
            a = np.abs(processed_sample) # the magnitude of frequency bin f at frame t
            # print("a", a)
            b = np.angle(processed_sample) # the phase of frequency bin f at frame t
            # print("b", b)
            c = a.flatten() + b.flatten() # #TODO find a good way to do this
            # print("c", c)

            processed_samples.append(c)
            sys.stdout.write("\rExtracting features %d%%" % floor((i + 1) * (100/len(samples))))
            sys.stdout.flush()
        print()
        return processed_samples