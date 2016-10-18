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

    def __init__(self, test_percentage=0.1, sampeling_rate=1000):
        self.sampeling_rate = sampeling_rate
        self.test_percentage = test_percentage

    def load_data(self):
        if MOCK_DATA:
            return self.load_mock_data()
        samples = []
        labels = []
        samples_test = []
        labels_test = []
        root = os.path.dirname(os.path.realpath(__file__)) + "/Data/AMTRecordings"
        for path, subdirs, files in os.walk(root):
            data_filenames = [name for name in files]
        included_filenames = []
        for filename in data_filenames:
            for allowed_filename in INCLUDED_VESSELS:
                if filename.startswith(allowed_filename):
                    included_filenames.append(filename)
                    break
        sample_every_n = int(self.test_percentage * 100)
        for i in range(len(included_filenames)):
            for sample_nr in range(SAMPLES_PR_FILE): # All files have to be at least 10 sec
                y, sr = librosa.load(root + "/" + included_filenames[i], sr=self.sampeling_rate, duration=SAMPLE_LENGTH, offset=sample_nr*SAMPLE_LENGTH)
                label = np.zeros(NR_OF_CLASSES)
                for j in range(NR_OF_CLASSES):
                    if included_filenames[i].startswith(INCLUDED_VESSELS[j]):
                        label[j] = 1
                        break
                # Add to either training or test set
                if ((SAMPLES_PR_FILE * i) + sample_nr) % sample_every_n == 0:
                    samples_test.append(y)
                    labels_test.append(label)
                else:
                    samples.append(y)
                    labels.append(label)
            sys.stdout.write("\rLoading data %d%%" % floor((i + 1) * (100/len(included_filenames))))
            sys.stdout.flush()
        print()
        #TODO pickle this to avoid loading each time
        return samples, labels, samples_test, labels_test

    def load_mock_data(self):

        # Autoencoder Make sure you dont test accuracy using a one-hot approach
        # samples = [np.array(self.bitfield(i, 3)) for i in range(8)]
        # labels = samples

        samples = [np.array([1, 0]) if i%2==0 else np.array([0, 1])  for i in range(20)] + [np.array([0, 0]) for i in range(10)]
        labels = [[1, 0] if i%2==0 else [0, 1]  for i in range(20)] + [[0, 1] for i in range(10)]
        '''
        samples = [[0 for i in range(SAMPELING_RATE//2)] + [1 for j in range(SAMPELING_RATE//2)] for k in range(NR_OF_CLASSES*10*SAMPLES_PR_FILE//2)]
        samples += [[1 for i in range(SAMPELING_RATE//2)] + [0 for j in range(SAMPELING_RATE//2)] for k in range(NR_OF_CLASSES*10*SAMPLES_PR_FILE//2)]
        labels = [[1, 0] for i in range(NR_OF_CLASSES*10*SAMPLES_PR_FILE//2)] + [[0, 1] for i in range(NR_OF_CLASSES*10*SAMPLES_PR_FILE//2)]
        '''
        samples_test = samples
        labels_test = labels

        return samples, labels, samples_test, labels_test

    def bitfield(self, n, k):
        bitarray = [int(digit) for digit in bin(n)[2:]]
        bitarray = ([0]*(k - len(bitarray))) + bitarray
        return bitarray



