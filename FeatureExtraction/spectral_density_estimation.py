import sys
from math import floor
from scipy import signal
from FeatureExtraction.feature_extractor_base import FeatureExtractorBase


class SpectralDensityEstimation(FeatureExtractorBase):
    def __init__(self, ):
        self.output_size = 0
        pass

    def extract_features(self, samples):
        processed_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            f, Pxx_den = signal.welch(sample)
            processed_samples.append(Pxx_den)
        sys.stdout.write("\rExtracting features %d%%" % floor((i + 1) * (100/len(samples))))
        sys.stdout.flush()
        print()
        self.output_size = len(processed_samples[0])
        return processed_samples