#!/usr/bin/python3

import cbor
import os
import sys
import pylab as pl

import matplotlib
import matplotlib.pyplot as pl
import matplotlib.legend_handler as lh

matplotlib.style.use('seaborn')
#matplotlib.rcParams['pdf.fonttype'] = 42

sys.path.append("..")
import utils

#
# This file plots the extraction time of features and transforms.
# The values are taken from the calc_energy_costs
#

NUM_SAMPLES = 15000
WINDOW_SIZE = utils.WINDOW_SIZE_SAMPLES 

FEATURES = [
    ("Empty loop", 3226),
    ("Jerk", 6016),
    ("Mean", 6770),
    ("Minumum", 6770),
    ("Maximum", 6770),
    ("Magnitude squared", 7656),
    ("Energy", 8436),
    ("Median-of-three", 8696),
    ("L1 norm", 8906),
    ("Stdev", 9190),
    ("Jerk + L1 norm", 12186),
    ("Jerk + Magn. sq.", 12500),
    ("First quartile", 16770),
    ("Median", 16770),
    ("Third quartile", 16770),
    ("Correlation", 17320),
    ("Inter-q. range", 18203),
    ("Entropy", 66926),
]

########################################

def plot(filename):
    fig, ax = pl.subplots()
    fig.set_size_inches((5, 2.1))

    labels= [u[0] for u in FEATURES]
    tm_usec = [u[1] for u in FEATURES]

    tm_per_window_sec = [u * WINDOW_SIZE / NUM_SAMPLES for u in tm_usec]

    ax.bar(range(len(labels)), tm_per_window_sec, label=labels)

    ax.set_xlabel("Extraction time per 128 samples")
    ax.set_ylabel("Microseconds")

#    range(len(labels)),
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)

    pl.savefig(filename,
               format='pdf',
#               bbox_extra_artists=(legend,),
               bbox_inches='tight')
    pl.close()

########################################

def main():
    plot("../results/feature-extraction-time.pdf")

########################################

    
if __name__ == '__main__':
    main()
    print("all done!")
