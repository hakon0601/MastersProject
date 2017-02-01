import pywt
import sys
from FeatureExtraction.feature_extractor_base import FeatureExtractorBase

from math import floor


class WaveletTransform(FeatureExtractorBase):
    def __init__(self, ):
        pass

    def extract_features(self, samples):
        processed_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            cA, cD = pywt.dwt(sample, 'db2')
            c = cA + cD # #TODO find a good way to do this

            processed_samples.append(c)
            sys.stdout.write("\rExtracting features %d%%" % floor((i + 1) * (100/len(samples))))
            sys.stdout.flush()
        print()
        return processed_samples