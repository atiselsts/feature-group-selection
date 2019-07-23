#!/usr/bin/python3

#
# File: check-distribution.py
# Description: This checks the class distribution in the input datasets (i.e. whether the dataset is balanced).
# Author: Atis Elsts, 2019
#

import os
import sys

sys.path.append("..")
import utils
from ml_config import *

#SUB = ["train_original", "train", "validation", "test"]

###########################################

def main():
    dataset = DEFAULT_DATASET
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    for sub in SUBSETS:
        dirname = os.path.join("..", "datasets", dataset, sub)
        short_sub = sub.replace("_original", "")
        filename = os.path.join(dirname, "y_{}.txt".format(short_sub))

        if not os.access(filename, os.R_OK):
            continue

        all_labels = {}
        print(filename)
        with open(filename, "r") as f:
            lines = [x.strip() for x in f.readlines() if x.strip() != ""]
            for x in lines:
                ix = int(x)
                all_labels[ix] = all_labels.get(ix, 0) + 1
                
        total = sum(all_labels.values())
        print("{}/{}:\t{} total".format(dataset, sub, total))
        for k in sorted(list(all_labels.keys())):
            v = all_labels[k]
            print("{}:\t{:2.2f}%".format(k, 100.0 * all_labels[k] / total))

###########################################

if __name__ == '__main__':
    main()
    print("all done!")
