'''
Python 3.4.3
librosa 0.4.3
Tensorflow 0.6.0, upgraded to 0.12.0
'''

from os import path

from math import ceil

DATA_PATH = path.dirname(path.realpath(__file__)) + "/Data"
LOG_PATH = "/tmp/tensorflow/mastersproject"



NR_OF_CLASSES = 7
INCLUDED_VESSELS = ["ferry", "nansen", "sejong", "speedboat", "svendborgmaersk", "tanker", "sub"]
# INCLUDED_VESSELS = ["speedboat", "tanker"]
# INCLUDED_VESSELS = ["ferry", "speedboat", "tanker", "sub"]
TEST_PERCENTAGE = 0.1
SAMPLING_RATE = 1024
NR_OF_FILES = 85
# 85 files in total
SAMPLES_PR_FILE = 100
SAMPLE_LENGTH = 1.0 # sec

FFT_WINDOW_SIZE = 2048
N_MFCC = 20

# Only one of these can be true at once
# Describes what part of the dataset is being used as test data
USE_WHOLE_FILE_AS_TEST = False # TODO fix the other two options for recurrent networks
USE_END_OF_FILE_AS_TEST = True
USE_RANDOM_SAMPLES_AS_TEST = False

# Recurrent NN
RELATED_STEPS = 20

NOISE_ENABLED = False
NR_OF_NOISY_SAMPLES_PR_SAMPLE = 2

RESET_PICKLE = False
MOCK_DATA = False
USE_PRELOADED_DATA = True

# if NR_OF_CLASSES != len(INCLUDED_VESSELS):
#     raise ValueError

# if SAMPLES_PR_FILE * SAMPLE_LENGTH > 10:
#     raise ValueError

BATCH_SIZE = 10
EPOCS = ceil(NR_OF_FILES * SAMPLES_PR_FILE / BATCH_SIZE) # To ensure all samples being used in training
EPOCS = 100
LEARNING_RATE = 0.0001
BIAS_ENABLED = False
DROPOUT_RATE = 0.9

ACTIVATION_FUNCTIONS = "2 2 2"
HIDDEN_LAYERS = "512 256 256"
CNN_FILTERS = "5 5 5"
CNN_CHANNELS = "32 64 64"
DCL_SIZE = 1024
PADDING = 2
POOLING_STRIDE = 2
POOLING_FILTER_SIZE = 2