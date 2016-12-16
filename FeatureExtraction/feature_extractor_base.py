
import abc

class FeatureExtractorBase():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def extract_features(self, data, show_spectrogram=False):
        return