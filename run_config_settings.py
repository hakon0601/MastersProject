'''
Python 3.4.3
librosa 0.4.3
Tensorflow 0.6.0
'''

from math import ceil

NR_OF_CLASSES = 7
INCLUDED_VESSELS = ["speedboat", "tanker"]
INCLUDED_VESSELS = ["ferry", "nansen", "sejong", "speedboat", "svendborgmaersk", "tanker", "sub"]
TEST_PERCENTAGE = 0.1
SAMPELING_RATE = 1000
NR_OF_FILES = 85
# 85 files in total
SAMPLES_PR_FILE = 100
SAMPLE_LENGTH = 0.5 # sec

MOCK_DATA = False
USE_PRELOADED_DATA = True

# if NR_OF_CLASSES != len(INCLUDED_VESSELS):
#     raise ValueError

# if SAMPLES_PR_FILE * SAMPLE_LENGTH > 10:
#     raise ValueError

BATCH_SIZE = 10
EPOCS = ceil(NR_OF_FILES * SAMPLES_PR_FILE / BATCH_SIZE) # To ensure all samples being used in training
EPOCS = 50
LEARNING_RATE = 0.001

ACTIVATION_FUNCTIONS = "1 1"
HIDDEN_LAYERS = "256 256"
