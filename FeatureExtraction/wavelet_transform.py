import pywt
import sys

from math import floor


class WaveletTransform():
    def __init__(self, ):
        pass

    def extract_features(self, samples, show_spectrogram=False):
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