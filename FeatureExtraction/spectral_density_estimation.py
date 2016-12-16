import sys
from math import floor
from scipy import signal


class SpectralDensityEstimation():
    def __init__(self, ):
        pass

    def extract_features(self, samples, show_spectrogram=False):
        processed_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            f, Pxx_den = signal.welch(sample)
            processed_samples.append(Pxx_den)
        sys.stdout.write("\rExtracting features %d%%" % floor((i + 1) * (100/len(samples))))
        sys.stdout.flush()
        print()
        return processed_samples