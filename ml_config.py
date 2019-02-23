#
# File: ml_config.py
# Description: defines the configuration for machine learning classifiers and feature selection algorithms.
# Author: Atis Elsts, 2019
#

DEFAULT_DATASET = "UCI HAR Dataset"

NUM_TREES = 100

# since from 0.0 to 1.0
W_ACCURACY = 500
RANDOM_ACCURACY = 0.4
# since from 10 to 1000, and the higher, the worse
W_ENERGY = -1

MAX_FEATURES_GREEDY = 10

USE_N_FOLD_CROSS_VALIDATION = False
NUM_VALIDATION_ITERATIONS = 5

# if cross-validation is not used: the number of trials on which the score is averaged
NUM_TRIALS = 1

def roundacc(acc):
    return int(round(acc))
