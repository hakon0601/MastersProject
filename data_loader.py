import os
import sys
import pickle
import librosa
import numpy as np
from random import shuffle, random
from math import floor

from run_config_settings import *
from tensorflow.examples.tutorials.mnist import input_data


class DataLoader():

    def __init__(self, test_percentage=0.1, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        self.test_percentage = test_percentage
        if RESET_PICKLE:
            dictionary = {}
            self.pickle_save(dictionary)

    def load_data(self, recurrent=False):
        if MOCK_DATA:
            return self.load_mock_data()
        data_dict = self.pickle_load()
        parameter_key = self.get_parameter_key(recurrent=recurrent)
        print("Parameter key", parameter_key)
        if USE_PRELOADED_DATA and parameter_key in data_dict:
            return data_dict[parameter_key]

        included_filenames = self.filter_only_included_vessels(self.get_data_filenames())
        shuffle(included_filenames) # Use one random file from each type as the testing set

        samples = []
        labels = []
        samples_test = []
        labels_test = []
        # sample_every_n = int(self.test_percentage * 100)
        files_used_for_testing = []
        for i in range(len(included_filenames)):
            samples_from_one_file, labels_from_one_file = self.load_samples_from_file(root=DATA_PATH, filename=included_filenames[i], recurrent=recurrent)
            if USE_WHOLE_FILE_AS_TEST:
                if included_filenames[i][:3] in files_used_for_testing: # The three first letters of a filename is type identificator
                    noisy_samples, noisy_labels = self.create_noisy_samples(samples_from_one_file, labels_from_one_file[0])
                    samples += samples_from_one_file + noisy_samples
                    labels += labels_from_one_file + noisy_labels
                else:
                    files_used_for_testing.append(included_filenames[i][:3])
                    samples_test += samples_from_one_file
                    labels_test += labels_from_one_file
            elif USE_RANDOM_SAMPLES_AS_TEST:
                for j in range(len(samples_from_one_file)):
                    if random() < TEST_PERCENTAGE:
                        samples_test.append(samples_from_one_file[j])
                        labels_test.append(labels_from_one_file[j])
                    else:
                        noisy_sample, noisy_label = self.create_noisy_samples([samples_from_one_file[j]], labels_from_one_file[j])
                        samples += [samples_from_one_file[j]] + noisy_sample
                        labels += [labels_from_one_file[j]] + noisy_label
            elif USE_END_OF_FILE_AS_TEST:
                nr_of_test_samples = ceil(len(samples_from_one_file) * TEST_PERCENTAGE)
                for j in range(len(samples_from_one_file)):
                    if j < len(samples_from_one_file) - nr_of_test_samples:
                        noisy_sample, noisy_label = self.create_noisy_samples([samples_from_one_file[j]], labels_from_one_file[j])
                        samples += [samples_from_one_file[j]] + noisy_sample
                        labels += [labels_from_one_file[j]] + noisy_label
                    else:
                        samples_test.append(samples_from_one_file[j])
                        labels_test.append(labels_from_one_file[j])

            sys.stdout.write("\rLoading data %d%%" % floor((i + 1) * (100/len(included_filenames))))
            sys.stdout.flush()
        print()
        # Pickle data to avoid loading each time
        data_dict[parameter_key] = [samples, labels, samples_test, labels_test]
        self.pickle_save(data_dict)
        return samples, labels, samples_test, labels_test

    def load_mock_data(self):
        return self.load_mnist()

    def load_mnist(self):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        samples = mnist.train.images
        labels = mnist.train.labels
        samples_test = mnist.test.images
        labels_test = mnist.test.labels
        return samples, labels, samples_test, labels_test

    @staticmethod
    def pickle_save(data_dict):
        # Adding the most recent data to the file
        with open(DATA_PATH + '/objs.pickle', 'wb') as f:
            pickle.dump(data_dict, f)

    @staticmethod
    def pickle_load():
        with open(DATA_PATH + '/objs.pickle', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def pickle_create():
        if not os.path.exists(DATA_PATH + '/objs.pickle'):
            open(DATA_PATH + '/objs.pickle', 'a').close()
            DataLoader.pickle_save(data_dict={})

    def get_parameter_key(self, recurrent):
        return str(NR_OF_CLASSES) + str(TEST_PERCENTAGE) + str(SAMPLING_RATE) + \
               str(SAMPLES_PR_FILE) + str(SAMPLE_LENGTH) + "rec=" + str(recurrent) + \
               "noise=" + str(NOISE_ENABLED) + str(NR_OF_NOISY_SAMPLES_PR_SAMPLE) + \
               str(USE_END_OF_FILE_AS_TEST) + str(USE_WHOLE_FILE_AS_TEST) + str(USE_RANDOM_SAMPLES_AS_TEST)

    def load_samples_from_file(self, root, filename, recurrent):
        samples = []
        labels = []
        duration = 10.0

        label = np.zeros(NR_OF_CLASSES)
        for j in range(NR_OF_CLASSES):
            if filename.startswith(INCLUDED_VESSELS[j]):
                label[j] = 1
                break

        if recurrent:
            # The length of each sample is the sample length * number of related steps
            offset = (duration - (SAMPLE_LENGTH * RELATED_STEPS)) / (SAMPLES_PR_FILE // RELATED_STEPS)
            for sample_nr in range(SAMPLES_PR_FILE // RELATED_STEPS):
                y, sr = librosa.load(root + "/AMTRecordings/" + filename, sr=self.sampling_rate, duration=SAMPLE_LENGTH * RELATED_STEPS, offset=sample_nr * offset)
                samples.append(y)
                labels.append(label)
        else:
            offset = (duration - SAMPLE_LENGTH) / max(1, (SAMPLES_PR_FILE - 1))
            for sample_nr in range(SAMPLES_PR_FILE):
                # May be better to load the file in its entirity and then split it into several samples
                y, sr = librosa.load(root + "/AMTRecordings/" + filename, sr=self.sampling_rate, duration=SAMPLE_LENGTH, offset=sample_nr * offset)
                samples.append(y)
                labels.append(label)

        return samples, labels

    def get_data_filenames(self):
        for path, subdirs, files in os.walk(DATA_PATH + "/AMTRecordings"):
            return [name for name in files]

    def filter_only_included_vessels(self, all_data_filenames):
        included_filenames = []
        for filename in all_data_filenames:
            for allowed_filename in INCLUDED_VESSELS:
                if filename.startswith(allowed_filename):
                    included_filenames.append(filename)
                    break
        return included_filenames

    def create_noisy_samples(self, samples_from_one_file, label):
        if not NOISE_ENABLED:
             return [], []
        for j in range(NR_OF_NOISY_SAMPLES_PR_SAMPLE):
            noisy_samples = []
            noisy_labels = []
            for i in range(len(samples_from_one_file)):
                std = np.std(samples_from_one_file[i])
                std = std*0.1
                noise = np.random.normal(np.mean(samples_from_one_file[i]), std, len(samples_from_one_file[i]))
                noisy_samples.append(samples_from_one_file[i] + noise)
                noisy_labels.append(label)
        return noisy_samples, noisy_labels
