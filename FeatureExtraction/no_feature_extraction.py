import sys
from math import floor

from FeatureExtraction.feature_extractor_base import FeatureExtractorBase
import librosa
import numpy as np


class NoFE(FeatureExtractorBase):
    def __init__(self):
        pass

    def extract_features(self, samples, show_spectrogram=False):
        return samples