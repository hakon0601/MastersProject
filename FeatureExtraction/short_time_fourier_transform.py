from FeatureExtraction.feature_extractor_base import FeatureExtractorBase
import librosa


class STFT(FeatureExtractorBase):
    def __init__(self):
        pass

    def extract_features(self, samples):
         processed_samples = []
         for sample in samples:
             processed_sample = librosa.stft(sample)
             processed_samples.append(processed_samples)
         print ("dsa")
         return processed_samples