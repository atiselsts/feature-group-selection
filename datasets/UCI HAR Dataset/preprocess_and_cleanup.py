#!/usr/bin/env python3

#
# File: UCI HAR Dataset/preprocess_and_cleanup.py
#
# Description: This script simply splits the train data in two parts.
# The larger initial train part is kept as train data.
# The smaller remaining part is put in a differen directory to be used as validation data.
#
# Author: Atis Elsts, 2019.
#

import sys
import os
import struct
import copy

INPUT_DIR = "train_original"
FILES = [
    "y_train.txt",
    "subject_train.txt",
    "Inertial Signals/total_acc_x_train.txt",
    "Inertial Signals/total_acc_y_train.txt",
    "Inertial Signals/total_acc_z_train.txt",
    "Inertial Signals/body_acc_x_train.txt",
    "Inertial Signals/body_acc_y_train.txt",
    "Inertial Signals/body_acc_z_train.txt",
    "Inertial Signals/body_gyro_x_train.txt",
    "Inertial Signals/body_gyro_y_train.txt",
    "Inertial Signals/body_gyro_z_train.txt",
]

#TRAIN_PROPORTION = 0.66
NUM_TRAIN_LINES = 4693

output_partitions = [
    ("train", NUM_TRAIN_LINES),
    ("validation", -1)
]

##########################################

def create_out_dir(outdir):
    try:
        os.mkdir(outdir)
    except Exception as ex:
        pass

###########################################

def separate_file(filename, to_keep):
    infilename = os.path.join(INPUT_DIR, filename)

    partname = output_partitions[0][0]
    outfilename1 = os.path.join(partname, filename)
    outfilename1 = outfilename1.replace("train", partname)

    partname = output_partitions[1][0]
    outfilename2 = os.path.join(partname, filename)
    outfilename2 = outfilename2.replace("train", partname)

    with open(infilename, "r") as f:
        lines = [l for l in f.readlines()]
        with open(outfilename1, "w") as outf:
            outf.write("".join(lines[:to_keep+1]))
        with open(outfilename2, "w") as outf:
            outf.write("".join(lines[to_keep+1:]))
    
###########################################

def main():
    for name, _ in output_partitions:
        create_out_dir(name)
        create_out_dir(os.path.join(name, "Inertial Signals"))
    filename = os.path.join(INPUT_DIR, FILES[0])
    with open(filename, "r") as f:
        n = len(f.readlines())
        to_keep = NUM_TRAIN_LINES
    for filename in FILES:
        separate_file(filename, to_keep)
        

###########################################

if __name__ == '__main__':
    main()
    print("all done!")
