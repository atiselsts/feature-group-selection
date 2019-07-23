#!/usr/bin/python3

#
# File: mutual_information.py
# Description: run the mutual information based feature selection algorithm.
# Author: Atis Elsts, 2019
#

import os
import numpy as np

from sklearn.metrics import mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score

import sys
sys.path.append("..")
sys.path.append("../energy-model")

import utils
import energy_model
from ml_config import *

import ml_state

NUM_BINS = 256

###########################################

class MIState(ml_state.State):
    def calc_MI(self, x, y, bins):
        c_xy = np.histogram2d(x, y, bins)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi

    def class_entropy(self, results, features, y, name, name_index, is_single = False):
        #print("class", name)

        selector = utils.select(self.names, self.groups, [name_index], self.do_subselection)
        r = 0
        for index in selector:
            feature = features[:,index]
            t = self.calc_MI(y, feature, NUM_BINS)
            #print("for feature with index", index, t)
            r += t

        results.append((r, name, name_index))


    def mi(self):
        results = []
        for i, grname in enumerate(self.groups):
            self.class_entropy(results, self.train, self.train_y, grname, i)

        results.sort(reverse=True)
        for r in results:
            print(r)

        return results
   

###########################################

def main():
    dataset = DEFAULT_DATASET
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    s = MIState()
    print("Loading...")
    s.load(dataset)
    print("Calculating mutual information...")
    r = s.mi()
    feature_indexes = []
    for i in range(len(r)):
        score, name, index = r[i]
        feature_indexes.append(index)
        score, av, at, e = s.combined_score(feature_indexes)
        if e >= s.energy_for_raw:
            break # too large energy, stop iterating
        print("best at", name, score, av, at, e)

###########################################

if __name__ == '__main__':
    main()
    print("all done!")
