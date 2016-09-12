import librosa
import numpy as np
from os import listdir

# Sample, Label
# Label: [Ferry, Nansen, Submarine, Sejong, Speedboat, SvendborgMaersk, Tanker]


class DataLoader():

    sampeling_rate = 1000
    included_vessels = ["ferry", "nansen", "sejong", "speedboat", "svendborgmaersk", "tanker", "sub"]
    classes = 7
    test_percentage = 0.1

    def __init__(self):
        self.samples = []
        self.labels = []


    def load_data(self):
        data_filenames = [f for f in listdir("data/AMTRecordings")]
        included_filenames = []
        for filename in data_filenames:
            for allowed_filename in self.included_vessels:
                if filename.startswith(allowed_filename):
                    included_filenames.append(filename)
                    break

        for filename in included_filenames:
            for t in range(10):
                y, sr = librosa.load("data/AMTRecordings/" + filename, sr=self.sampeling_rate, duration=1, offset=t)
                self.samples.append(y)
                label = np.zeros(self.classes)
                for i in range(len(self.included_vessels)):
                    if filename.startswith(self.included_vessels[i]):
                        label[i] = 1
                        break
                self.labels.append(label)

        test_cut_nr = int(len(self.samples)*(1 - self.test_percentage))
        samples_test = self.samples[test_cut_nr:]
        labels_test = self.labels[test_cut_nr:]
        self.samples = self.samples[:test_cut_nr]
        self.labels = self.labels[:test_cut_nr]
        print("Data loaded")
        return self.samples, self.labels, samples_test, labels_test
