import os
import pickle
import librosa
import numpy as np
from os import listdir
from math import floor

import sys

from run_config_settings import *
from tensorflow.examples.tutorials.mnist import input_data

# Sample, Label
# Label: [Ferry, Nansen, Submarine, Sejong, Speedboat, SvendborgMaersk, Tanker]

class DataLoader():

    def __init__(self, test_percentage=0.1, sampeling_rate=1000):
        self.sampeling_rate = sampeling_rate
        self.test_percentage = test_percentage

    def load_data(self):
        if MOCK_DATA:
            return self.load_mock_data()
        data_dict = self.pickle_load()
        parameter_key = self.get_parameter_key()
        print("Parameter key", parameter_key)
        if USE_PRELOADED_DATA:
            if parameter_key in data_dict:
                return data_dict[parameter_key]
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
            # y, sr = librosa.load(root + "/" + included_filenames[i])
            # duration = librosa.get_duration(y=y, sr=sr)
            duration = 10.0
            offset = (duration - SAMPLE_LENGTH) / (SAMPLES_PR_FILE - 1)
            for sample_nr in range(SAMPLES_PR_FILE): # All files have to be at least 10 sec
                y, sr = librosa.load(root + "/" + included_filenames[i], sr=self.sampeling_rate, duration=SAMPLE_LENGTH, offset=sample_nr*offset)
                label = np.zeros(NR_OF_CLASSES)
                for j in range(NR_OF_CLASSES):
                    if included_filenames[i].startswith(INCLUDED_VESSELS[j]):
                        label[j] = 1
                        break
                # Add to either training or test set
                # if ((SAMPLES_PR_FILE * i) + sample_nr) % sample_every_n == 0:
                if (i % 10) == 0:
                    samples_test.append(y)
                    labels_test.append(label)
                else:
                    samples.append(y)
                    labels.append(label)
            sys.stdout.write("\rLoading data %d%%" % floor((i + 1) * (100/len(included_filenames))))
            sys.stdout.flush()
        print()
        # Pickle data to avoid loading each time
        data_dict[parameter_key] = [samples, labels, samples_test, labels_test]
        self.pickle_save(data_dict)
        return samples, labels, samples_test, labels_test

    def load_mock_data(self):

        return self.load_mnist()

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

    def load_mnist(self):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        samples = mnist.train.images
        labels = mnist.train.labels
        samples_test = mnist.test.images
        labels_test = mnist.test.labels
        return samples, labels, samples_test, labels_test


    def bitfield(self, n, k):
        bitarray = [int(digit) for digit in bin(n)[2:]]
        bitarray = ([0]*(k - len(bitarray))) + bitarray
        return bitarray

    def pickle_save(self, data_dict):
        # Adding the most recent data to the file
        with open('objs.pickle', 'wb') as f:
            pickle.dump(data_dict, f)

    def pickle_load(self):
        with open('objs.pickle', 'rb') as f:
            return pickle.load(f)

    def get_parameter_key(self):
        return str(NR_OF_CLASSES) + str(TEST_PERCENTAGE) + str(SAMPELING_RATE) + str(SAMPLES_PR_FILE) + str(SAMPLE_LENGTH)





