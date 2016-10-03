from math import ceil

TEST_PERCENTAGE = 0.1
SAMPELING_RATE = 1000
NR_OF_FILES = 85
SAMPLES_PR_FILE = 5
# 85 files in total
SAMPLE_LENGTH = 1 # sec
if SAMPLES_PR_FILE * SAMPLE_LENGTH > 10:
    raise ValueError

BATCH_SIZE = 10
TRAINING_ITERATIONS = 1
TRAINING_ITERATIONS = ceil(NR_OF_FILES * SAMPLES_PR_FILE / BATCH_SIZE) # To ensure all samples being used in training
