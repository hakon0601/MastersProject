'''
Python 3.4.3
librosa 0.4.3
Tensorflow 0.6.0
'''

from math import ceil

NR_OF_CLASSES = 2
INCLUDED_VESSELS = ["ferry", "nansen", "sejong", "speedboat", "svendborgmaersk", "tanker", "sub"]
INCLUDED_VESSELS = ["speedboat", "tanker"]
TEST_PERCENTAGE = 0.1
SAMPELING_RATE = 1000
NR_OF_FILES = 85
SAMPLES_PR_FILE = 5
# 85 files in total
SAMPLE_LENGTH = 1 # sec

if NR_OF_CLASSES != len(INCLUDED_VESSELS):
    raise ValueError

if SAMPLES_PR_FILE * SAMPLE_LENGTH > 10:
    raise ValueError

BATCH_SIZE = 1
TRAINING_ITERATIONS = ceil(NR_OF_FILES * SAMPLES_PR_FILE / BATCH_SIZE) # To ensure all samples being used in training
TRAINING_ITERATIONS = 100
