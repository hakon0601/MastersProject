import sys
from math import floor

from FeatureExtraction.feature_extractor_base import FeatureExtractorBase
import librosa
import numpy as np
import matplotlib.pyplot as plt


class STFT(FeatureExtractorBase):
    def __init__(self, fft_window_size=2048):
        self.fft_window_size = fft_window_size

    def extract_features(self, samples, show_spectrogram=False):
        if show_spectrogram:
            # TODO create a reasonable spectrogram
            plt.figure(figsize=(12, 8))
            D = librosa.logamplitude(np.abs(librosa.stft(samples[0]))**2, ref_power=np.max)
            plt.subplot(4, 2, 1)
            librosa.display.specshow(D, y_axis='linear')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Linear-frequency power spectrogram')
            plt.show()
            print("sdads")
        processed_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            processed_sample = librosa.stft(sample, self.fft_window_size) # Creates an array of touples, complex values? fix #TODO time-freq?
            a = np.abs(processed_sample) # the magnitude of frequency bin f at frame t
            b = np.angle(processed_sample) # the phase of frequency bin f at frame t
            c = a.flatten() + b.flatten() # #TODO find a good way to do this

            processed_samples.append(c)
            sys.stdout.write("\rExtracting features %d%%" % floor((i + 1) * (100/len(samples))))
            sys.stdout.flush()
        print()
        return processed_samples