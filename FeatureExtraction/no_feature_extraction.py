from FeatureExtraction.feature_extractor_base import FeatureExtractorBase


class NoFE(FeatureExtractorBase):
    def __init__(self):
        self.output_size = 0
        pass

    def extract_features(self, samples):
        self.output_size = len(samples[0])
        return samples