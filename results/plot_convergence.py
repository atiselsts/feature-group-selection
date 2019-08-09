#!/usr/bin/python3

import os
import sys
import numpy as np
from collections import Counter

import matplotlib
import matplotlib.pyplot as pl
import matplotlib.legend_handler as lh

matplotlib.style.use('seaborn')
#matplotlib.rcParams['pdf.fonttype'] = 42

sys.path.append("..")
sys.path.append("../energy-model")
import utils
import energy_model

###########################################

SHOW_RAW = False

CATEGORIES = []

###########################################

def plot(data, filename):
    pl.figure(figsize=(4,3))
    for d, label in data:
        pl.plot(range(1, len(d) + 1), d, label=label)
    pl.xlabel("Iteration")
    pl.ylabel("Combined score")
    pl.ylim(0, 450)
    pl.legend()
    pl.tight_layout()
    pl.savefig(filename)
    pl.close()

###########################################

def read_data(dataset):
    data = []
    filename = dataset + "_convergence.log"
    with open(filename, "r") as f:
        for line in f:
            #Best:  Particle with #features=6 accuracy=0.8745/0.9097 energy=12.4301 score=424.5699 features=[tTotalAcc-q75(),tTotalAccJerk-energy(),tTotalAccJerkL1Norm-min(),tTotalAccJerkMagSq-min(),tTotalAccMagSq-max(),tTotalAccMagSq-std()]
            if "Best:" in line:
                fields = line.split()
                if len(fields) > 4:
                    acc = fields[4].split("=")[1]
                    av, at = [float(x) for x in acc.split("/")]
                    score = float(fields[6].split("=")[1])
                    data.append(score)
    return data

###########################################

def main():
    all_data = []
    for dataset in utils.ALL_DATASETS_SHORT:
        data = read_data(dataset)
        all_data.append((data, dataset))
    plot(all_data, "convergence.pdf")

###########################################
    
if __name__ == '__main__':
    main()
    print("all done!")
