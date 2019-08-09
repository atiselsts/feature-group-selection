#!/usr/bin/env python3

#
# File: PAMAP2/preprocess_and_cleanup.py
#
# The purpose of this script is to put the PAMAP2 in a format similar to the HAR dataset.
#
# Steps:
#
# * extract the relevant raw data (columns 1, 7, 8, 9) from the .dat files
#
# * separate the raw data in 128-sample vectors, each window with a specific label
#
# * sort the windows according to classes
#
# * separate the resulting data data in train, validation, and test sets according to 50/25/25% proprortions
#   (choose the rows in balanced, random fashion)
#
# Author: Atis Elsts, 2019
#

import sys
import os
import copy
import random
from collections import Counter
import math

SELF_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append("../..")
import utils
from ml_config import *

INPUT_DIR = "./Protocol"

INPUTS = [
    "subject101.dat",
    "subject102.dat",
    "subject103.dat",
    "subject104.dat",
    "subject105.dat",
    "subject106.dat",
    "subject107.dat",
    "subject108.dat",
    "subject109.dat",
]

#
# Expected input file format example (space-separated, first columns only).
# Columns indexed 1, 7, 8, and 9 should be extracted.
#
# time activity heartrate hand-temperature hand-accel-hires-x hand-accel-hires-y hand-accel-hires-z hand-accel-lowres-x hand-accel-lowres-y hand-accel-lowres-z ...
# 15.47 0 141 24.75 -9.98278 2.61759 2.46038 -10.2341 2.69055 2.45977 ...

OUT_DIR = "Inertial Signals"

AXES = ["x", "y", "z"]

SAMPLING_RATE_HZ = 100

PER_SAMPLE = 1.0 / SAMPLING_RATE_HZ

WINDOW_SIZE_SAMPLES = utils.WINDOW_SIZE_SAMPLES
WINDOW_SIZE_SECONDS = WINDOW_SIZE_SAMPLES / SAMPLING_RATE_HZ

#TRAIN_PROPORTION = 0.5
#VALIDATION_PROPORTION = 0.25
# the test samples take the rest!

# meters per second squared
SCALING_FACTOR_ONE_G = 9.80665 

##########################################

def create_out_dir(outdir):
    try:
        os.mkdir(outdir)
    except Exception as ex:
        pass

##########################################

def load_file(filename):
    activities = []
    x = []
    y = []
    z = []

    with open(filename, "r") as f:
        for line in f.readlines()[1:]:
            d = line.strip().split()
            activities.append(int(d[1]))
            x.append(float(d[7]) / SCALING_FACTOR_ONE_G)
            y.append(float(d[8]) / SCALING_FACTOR_ONE_G)
            z.append(float(d[9]) / SCALING_FACTOR_ONE_G)

            # missing data
            if math.isnan(x[-1]):
                x[-1] = x[-2]
            if math.isnan(y[-1]):
                y[-1] = y[-2]
            if math.isnan(z[-1]):
                z[-1] = z[-2]

    # round to whole windows
    rounded_size = len(activities) // WINDOW_SIZE_SAMPLES * WINDOW_SIZE_SAMPLES
    activities = activities[:rounded_size]
    x = x[:rounded_size]
    y = y[:rounded_size]
    z = z[:rounded_size]

    return activities, x, y, z
                      
##########################################

def process():
    subjects = []
    activities = []
    data = {"x" : [], "y" : [], "z" : []}

    for inputname in INPUTS:
        print("Loading input file " + inputname)
        subject = int(inputname[7:10])

        input_filename = os.path.join(INPUT_DIR, inputname)
        tactivities, tx, ty, tz = load_file(input_filename)

        subjects += [subject] * len(tactivities)
        activities += tactivities
        data["x"] += tx
        data["y"] += ty
        data["z"] += tz

    MIN_COUNT = 2 * WINDOW_SIZE_SAMPLES / 3.0

    print("Separating in classes")
    per_label = {}
    i = 0
    while i + WINDOW_SIZE_SAMPLES <= len(activities):
        slice = activities[i:i+WINDOW_SIZE_SAMPLES]
        subject = subjects[i] # as all slice must have the same subject
        value, count = Counter(slice).most_common()[0]
        if count >= MIN_COUNT and value != 0:
            if value not in per_label:
                per_label[value] = []
            # just remember the start of the data and the subject
            per_label[value].append((i, subject))
        i += WINDOW_SIZE_SAMPLES

    print("Separating in train / validation / test...")
    per_sub_per_label = {}
    for subset in SUBSETS:
        per_sub_per_label[subset] = {}

    for label in per_label:
        random.shuffle(per_label[label])
        n = len(per_label[label])
        train_n = n // 2
        validation_n = n // 4
        test_n = n - (validation_n + train_n)

        per_sub_per_label["train"][label] = per_label[label][:train_n]
        per_sub_per_label["validation"][label] = per_label[label][train_n:(train_n + validation_n)]
        per_sub_per_label["test"][label] = per_label[label][(train_n + validation_n):]


    for sub in SUBSETS:
        create_out_dir(os.path.join("./", sub))
        create_out_dir(os.path.join("./", sub, OUT_DIR))

        order = sorted(per_sub_per_label[sub].keys())

        print("writing labels for", sub)
        filename = os.path.join("./", sub, "y_{}.txt".format(sub))
        with open(filename, "w") as outf:
            to_write = []
            for label in order:
                for index, subject in per_sub_per_label[sub][label]:
                    to_write.append(str(label))
            outf.write("\n".join(to_write) + "\n")

        print("writing subjects for", sub)
        filename = os.path.join("./", sub, "subject_{}.txt".format(sub))
        with open(filename, "w") as outf:
            to_write = []
            for label in order:
                for index, subject in per_sub_per_label[sub][label]:
                    to_write.append(str(subject))
            outf.write("\n".join(to_write) + "\n")

        print("writing data for", sub)
        # dump three files (for each axis)
        for i, a in enumerate(AXES):
            filename = os.path.join("./", sub, OUT_DIR, "total_acc_{}_{}.txt".format(a, sub))
            with open(filename, "w") as outf:
                for label in order:
                    for index, subject in per_sub_per_label[sub][label]:
                        slice = data[a][index:index+WINDOW_SIZE_SAMPLES]
                        sslice = ["{:e}".format(u) for u in slice]
                        outf.write(" ".join(sslice) + "\n")


###########################################

def main():
    process()

###########################################

if __name__ == '__main__':
    main()
    print("all done!")
