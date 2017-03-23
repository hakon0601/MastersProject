import sys
import matplotlib.pyplot as plt
from FeatureExtraction.feature_extractor_base import FeatureExtractorBase
from run_config_settings import *
from math import floor
import numpy as np
import random

plt.ioff()

class Spectrogram(FeatureExtractorBase):
    def __init__(self, ):
        self.output_size = 0
        pass

    def extract_features(self, samples):
        processed_samples = []
        for i in range(len(samples)):
            A = plt.specgram(samples[i], NFFT=2**6 - 1, Fs=SAMPLING_RATE, noverlap=0) #32*16
            self.output_size = (len(A[0]), len(A[0][0]))
            processed_samples.append(A[0].flatten())

            sys.stdout.write("\rExtracting features %d%%" % floor((i + 1) * (100/len(samples))))
            sys.stdout.flush()
        print()
        return processed_samples

    def save_specgrams(self, samples, labels, select_random=False, nr_of_images=30):
        for i in range(nr_of_images):
            if select_random:
                r = random.randint(0, len(samples))
            else:
                r = i
            label = INCLUDED_VESSELS[np.argmax(labels[r])]
            plt.figure()
            A = plt.specgram(samples[r], NFFT=2**6 - 1, Fs=SAMPLING_RATE, noverlap=0)
            plt.savefig("spec_" + label + "_" + str(i) + ".png")
            plt.close("all")
