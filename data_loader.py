import os

import librosa
import numpy as np
from os import listdir
from math import floor

import sys

from run_config_settings import *

# Sample, Label
# Label: [Ferry, Nansen, Submarine, Sejong, Speedboat, SvendborgMaersk, Tanker]


class DataLoader():

    included_vessels = ["ferry", "nansen", "sejong", "speedboat", "svendborgmaersk", "tanker", "sub"]
    classes = 7

    def __init__(self, test_percentage=0.1, sampeling_rate=1000):
        self.sampeling_rate = sampeling_rate
        self.test_percentage = test_percentage


    def load_data(self):
        samples = []
        labels = []
        samples_test = []
        labels_test = []
        root = os.path.dirname(os.path.realpath(__file__)) + "/Data/AMTRecordings"
        for path, subdirs, files in os.walk(root):
            data_filenames = [name for name in files]
        included_filenames = []
        for filename in data_filenames:
            for allowed_filename in self.included_vessels:
                if filename.startswith(allowed_filename):
                    included_filenames.append(filename)
                    break
        sample_every_n = int(self.test_percentage * 100)
        for i in range(len(included_filenames)):
            for sample_nr in range(SAMPLES_PR_FILE): # All files have to be at least 10 sec
                y, sr = librosa.load(root + "/" + included_filenames[i], sr=self.sampeling_rate, duration=SAMPLE_LENGTH, offset=sample_nr*SAMPLE_LENGTH)
                label = np.zeros(self.classes)
                for j in range(len(self.included_vessels)):
                    if included_filenames[i].startswith(self.included_vessels[j]):
                        label[j] = 1
                        break
                if ((SAMPLES_PR_FILE * i) + sample_nr) % sample_every_n == 0:
                    samples_test.append(y)
                    labels_test.append(label)
                else:
                    samples.append(y)
                    labels.append(label)
            sys.stdout.write("\rLoading data %d%%" % floor((i + 1) * (100/len(included_filenames))))
            sys.stdout.flush()
        print()
        return samples, labels, samples_test, labels_test
