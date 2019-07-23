#!/usr/bin/env python3

#
# File: SPHERE/preprocess_and_cleanup.py
#
# The purpose of this script is to prepare the SPHERE Challege dataset in a format similar to the HAR dataset.
#
# Steps:
#
# * separate the raw data in 128-sample vectors, each window with a specific label
#
# * label the windows where there is less than 2/3 majority of a single activity with "UNKNOWN"
#
# * filter out windows that do not have labels in the core 3 activities;
#   save the windows that passed the filter in a file.
#
# * separate the *filtered* data in train, validation, and test sets according to 50/25/25% proprortions
#   (choose the rows in balanced, random fashion)
#
# Author: Atis Elsts, 2018
#

import sys
import os
import struct
import copy
import random

SELF_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append("../..")
import utils
from labels import LABEL_TO_CODE, ACTIVITY_SYNONYMS

INPUT_DIR = "./open_sphere_challenge_data"

INPUTS = [
    "00001",
    "00002",
    "00003",
    "00004",
    "00005",
    "00007",
]

ONLY_LABELS = ["SITTING", "STANDING", "LAYING"]
ONLY_LABEL_CODES = [str(LABEL_TO_CODE[l]) for l in ONLY_LABELS]

#
# Expected input file format example (comma-separated):
#
# t,x,y,z,Kitchen_AP,Lounge_AP,Upstairs_AP,Study_AP
# 0.017856,0.944,-0.28,0.152,-93.0,-95.0,-79.0,
#

OUT_DIR = "Inertial Signals"

AXES = ["x", "y", "z"]

NUM_ANNOTATIONS = 2

# extract data about this much seconds; it should match exactly 235 non-overlapping windows
NUM_SECONDS = 1504

SAMPLING_RATE_HZ = 20

PER_SAMPLE = 1.0 / SAMPLING_RATE_HZ

WINDOW_SIZE_SAMPLES = utils.WINDOW_SIZE_SAMPLES
WINDOW_SIZE_SECONDS = WINDOW_SIZE_SAMPLES / SAMPLING_RATE_HZ

# window_size == 128 samples, but windows are that there are partially overlapping (50% overlapping)
# windows are 6.4 seconds long, but the interval between windows is 3.2 seconds.
NUM_WINDOWS = 2 * int(round(NUM_SECONDS / WINDOW_SIZE_SECONDS))

NUM_SAMPLES = (NUM_WINDOWS * WINDOW_SIZE_SAMPLES) // 2

TRAIN_PROPORTION = 0.5
VALIDATION_PROPORTION = 0.25
# the test samples take the rest!


##########################################

def create_out_dir(outdir):
    try:
        os.mkdir(outdir)
    except Exception as ex:
        pass

##########################################

def load_file(filename):
    with open(filename, "r") as f:
        tdata = []
        sdata = []
        oldt = None
        oldv = [0, 0, 0]
        total = 0
        totalmissing = 0
        for line in f.readlines()[1:]:
            d = line.strip().split(",")
            t = float(d[0])
            x = d[1]
            y = d[2]
            z = d[3]
            v = list(map(float, [x, y, z]))

            if oldt is None:
                oldt = t - 0.05
            nummissing = 0
            oldoldt = oldt
            while oldt + 0.05 + 0.002 < t:
                #print("missing sample before t=", t, oldt)
                nummissing += 1
                total += 1
                totalmissing += 1
                oldt += 0.05
                sdata.append(oldv)
                tdata.append(oldt)


#            if nummissing:
#                print("missing before t=", t, " num=", nummissing, " old=", (oldoldt + 0.05))

            total += 1

            oldt = t
            oldv = copy.copy(v)
            sdata.append(v) #", ".join(map(str, v)))
            tdata.append(t)


        print("pdr: {:0.6f}".format(100.0 - 100.0 * float(totalmissing) / total))
        print("t=", t)

        return sdata, tdata

##########################################

def dump_activities(activities, window_period, window_size, filename):
    with open(filename, "w") as f:
        with open(filename + ".debug", "w") as debugf:
            for second in range(NUM_SECONDS):
                index = second * window_period
                end = index + window_size
                time_seconds = index / SAMPLING_RATE_HZ
                if end > len(activities):
                    print("detected end of activities at", time_seconds, "seconds")
                    break

                counts = {a : 0 for a in LABEL_TO_CODE}

                for i in range(index, end):
                    counts[activities[i]] += 1

                by_count = sorted(list(counts.items()), key=lambda x : x[1], reverse=True)
                best_a, best_count = by_count[0]
                # accept if more than 2/3 of the previous window has this activity
                if best_count <= window_size * 2 / 3:
                    best_a = "UNKNOWN"

                if best_a not in LABEL_TO_CODE:
                    print("Unknow activity ", best_a)
                    best_a = "UNKNOWN"

                best_a_code = LABEL_TO_CODE.get(best_a)
                debugf.write("{:6.1f}:\t{} {}\n".format(time_seconds, best_a, best_a_code))
                f.write("{}\n".format(best_a_code))

##########################################

def median(a, b, c):
    if a > b:
        if b > c:
            return b # a, b, c
        if a > c:
            return c # a, c, b
        return a # c, a, b
    else: # a <= b
        if a > c:
            return a # b, a, c
        if b > c:
            return c # b, c, a
        return b # c, b, a

def median_filter(sdata):
    print("Applying median filter")
    result = []
    result.append(sdata[0])
    for i in range(1, len(sdata) - 1):
        x = median(sdata[i-1][0], sdata[i][0], sdata[i+1][0])
        y = median(sdata[i-1][1], sdata[i][1], sdata[i+1][1])
        z = median(sdata[i-1][2], sdata[i][2], sdata[i+1][2])
        result.append((x, y, z))
    result.append(sdata[-1])          
    return result
                        
##########################################

def merge_annotations(dirname):
    num_annotations = 0
    input_dirname = os.path.join(INPUT_DIR, dirname)
    output_dirname = input_dirname

    for i in range(NUM_ANNOTATIONS):
        filename = os.path.join(input_dirname, "annotations_{}.csv".format(i))
        if os.access(filename, os.R_OK):
            num_annotations += 1
        else:
            break

    merged_annotations = []
    current_annotations = ["UNKNOWN"] * num_annotations
    current_merged_annotation = None
    all_annotations = []
    time_per_activity = { a : 0 for a in LABEL_TO_CODE }

    for i in range(num_annotations):
        filename = os.path.join(input_dirname, "annotations_{}.csv".format(i))
        with open(filename, "r") as f:
            for line in f.readlines()[1:]:
                fields = line.strip().split(",")
                try:
                    t1 = float(fields[0])
                    t2 = float(fields[1])
                    activity = fields[2]
                    if activity not in LABEL_TO_CODE:
                        # try to find synonym for this
                        activity = ACTIVITY_SYNONYMS.get(activity)
                        if activity is None:
                            # failed; it's something else
                            activity = "UNKNOWN"

                    all_annotations.append((t1, t2, activity, i))
                except Exception as ex:
                    print("Exception " + str(ex) + " in line: " + line)

    all_annotations.sort()
    for t1, t2, activity, i in all_annotations:

        if current_merged_annotation:
            # end it right here
            current_merged_annotation[1] = min(t1, current_merged_annotation[1])
            tm = current_merged_annotation[1] - current_merged_annotation[0]
            # just a heuristic that should not affect the output:
            # ignore activities shorter than one second
            if tm >= 1.0:
                merged_annotations.append(current_merged_annotation)
                #print("{} {:0.2f}".format(current_merged_annotation[2], tm))
                time_per_activity[current_merged_annotation[2]] += tm
            current_merged_annotation = None

        current_annotations[i] = activity

        all_same = activity != "UNKNOWN"
        for j in range(num_annotations):
            if current_annotations[j] != current_annotations[i]:
                all_same = False

        if all_same:
            current_merged_annotation = [t1, t2, activity]

    if current_merged_annotation:
        tm = current_merged_annotation[1] - current_merged_annotation[0]
        if tm >= 1.0: # see comment above
            merged_annotations.append(current_merged_annotation)
            #print("{} {:0.2f}".format(current_merged_annotation[2], tm))
            time_per_activity[current_merged_annotation[2]] += tm
        current_merged_annotation = None

    # assign each accelerometer sample a specific activity
    # (or "UNKNOWN" if the annotator disagree)
    activities = ["UNKNOWN"] * NUM_SAMPLES
    for t1, t2, activity in merged_annotations:
        start = int(round(t1 * SAMPLING_RATE_HZ))
        end = int(round(t2 * SAMPLING_RATE_HZ)) + 1
        for i in range(start, end):
            if i >= len(activities):
                break
            activities[i] = activity

    filename = os.path.join(output_dirname, "labels.csv")
    window_period = WINDOW_SIZE_SAMPLES // 2 # each 64 samples in case of 128 sample window
    dump_activities(activities, window_period, WINDOW_SIZE_SAMPLES, filename)

##########################################

def pick_data(start_t, timestamped_data):
    num_samples = 0
    r = []
    for t, el in timestamped_data:
        if t < start_t:
            continue
        r.append(el)
        num_samples += 1
        if num_samples >= WINDOW_SIZE_SAMPLES:
            break
    return r

##########################################

def prepare_format():
    for dirname in INPUTS:
        print("Loading files from " + dirname)

        merge_annotations(dirname)

        input_filename = os.path.join(INPUT_DIR, dirname, "acceleration_corrected.csv")

        sdata, tdata = load_file(input_filename)

        # XXX: do not apply the median filter now: it is applied later, at the feature extraction stage
        # sdata = median_filter(sdata)

        windows = []

        timestamped_data = list(zip(tdata, sdata))

        for i in range(NUM_WINDOWS - 1):
            window_start = i * (WINDOW_SIZE_SECONDS / 2)
            data = pick_data(window_start, timestamped_data)
            windows.append(data)

        # dump three files (for each axis)
        for i in range(3):
            axis = AXES[i]
            outfilename = os.path.join(INPUT_DIR, dirname, axis + ".csv")
            with open(outfilename, "w") as outf:
                for j, window in enumerate(windows):
                    window_start = j * (WINDOW_SIZE_SECONDS / 2)
                    if len(window) != WINDOW_SIZE_SAMPLES:
                        print("half empty window at ", window_start)
                        break
                    lst = ["{:e}".format(el[i]) for el in window]
                    #outf.write("{:6.1f}: ".format(window_start))
                    outf.write(" ".join(lst) + "\n")


###########################################

def split_in_datasets():

    classes = [
        ("train", TRAIN_PROPORTION),
        ("validation", VALIDATION_PROPORTION),
        ("test", -1),
    ]
    print("classes", classes)

    all_data = []

    print("Rereading cleaned up input data...")
    for dirname in INPUTS:
        num_labels_per_class = {}
        subject = int(dirname.strip("0"))

        filename = os.path.join(INPUT_DIR, dirname, "labels.csv")
        with open(filename, "r") as f:
            lines = [l.strip() for l in f.readlines()]
            labels =  [l for l in lines if l != ""]

        windows = {}
        for a in AXES:
            filename = os.path.join(INPUT_DIR, dirname, a + ".csv")
            with open(filename, "r") as f:
                lines = [l.strip() for l in f.readlines()]
                windows[a] = [l for l in lines if l != ""]

                # all the axis should have the same amount of data and it should also match the labels
                assert len(labels) == len(windows[a])

        for i in range(len(labels)):
            label = labels[i]
            all_data.append((subject, label, windows["x"][i], windows["y"][i], windows["z"][i]))

    print("Separating and filtering data...")
    filtered_data = [x for x in all_data if x[1] in ONLY_LABEL_CODES]

    partitions = {}
    for partname, _ in classes:
        partitions[partname] = []

    per_label = {}
    for label in ONLY_LABEL_CODES:
        per_label[label] = [x for x in all_data if x[1] == label]
        n = len(per_label[label])
        # shuffle the data randomly, to avoid putting different subjects in the test set than in the train set
        random.shuffle(per_label[label])

        start = 0
        for partname, proportion in classes:
            if proportion == -1:
                end = n
            else:
                end = start + int(proportion * n)
            data = per_label[label][start:end]
            #print("add {} {} : {} {}".format(label, partname, start, end))
            partitions[partname] += data
            start = end

    print("Creating output files...")
    for partname, _ in classes:
        create_out_dir(os.path.join("./", partname))
        create_out_dir(os.path.join("./", partname, OUT_DIR))

        filename = os.path.join("./", partname, "subject_{}.txt".format(partname))
        with open(filename, "w") as outf:
            subjects = [str(u[0]) for u in partitions[partname]]
            outf.write("\n".join(subjects) + "\n")

        filename = os.path.join("./", partname, "y_{}.txt".format(partname))
        with open(filename, "w") as outf:
            labels = [u[1] for u in partitions[partname]]
            outf.write("\n".join(labels) + "\n")

        for i, a in enumerate(AXES):
            filename = os.path.join("./", partname, OUT_DIR, "total_acc_{}_{}.txt".format(a, partname))
            with open(filename, "w") as outf:
                windows = [u[i + 2] for u in partitions[partname]]
                outf.write("\n".join(windows) + "\n")


###########################################

def main():
    prepare_format()
    split_in_datasets()

###########################################

if __name__ == '__main__':
    main()
    print("all done!")
