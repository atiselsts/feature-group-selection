#!/usr/bin/python3 -u

#
# File: greedy_algorithms.py
# Description: run the greedy feature selection algorithms. Two scoring versions are implemented: combined (used in the paper) and accuracy (F1 score) only.
# Author: Atis Elsts, 2019
#

import os
import numpy as np
import copy

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

###########################################
 
class GreedyState(ml_state.State):
    def greedy(self):
        # start iterating
        self.greedy_iteration([], float("-inf"))

    def greedy_iteration(self, used_features, prev_best):
        best_score = float("-inf")
        best_f = -1
        best_av = None
        best_at = None
        best_e = None

        for f in range(self.num_features):
            if f in used_features:
                continue

            #print(self.groups[f])

            updated_features = copy.copy(used_features)
            updated_features.append(f)

            if self.use_accuracy_only:
                av, at = self.eval_accuracy(updated_features)
                score = av
                e = None
            else:
                score, av, at, e = self.combined_score(updated_features)

            if score > best_score:
                best_score = score
                best_f = f
                best_av = av
                best_at = at
                best_e = e

        if best_f == -1:
            # not found any features to use
            return

        if best_e >= self.energy_for_raw:
            # energy gets too large
            print("stopping: spent more energy than for raw data Tx {:.4f} vs {:.4f}".format(
                  best_e, self.energy_for_raw))
            return

        print("best at", self.groups[best_f], best_score, best_av, best_at, best_e)
        updated_features = copy.copy(used_features)
        updated_features.append(best_f)
        print("one level deeper, used=", [self.groups[x] for x in updated_features])
        self.greedy_iteration(updated_features, best_score)

###########################################

def main():
    dataset = DEFAULT_DATASET
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    s = GreedyState()
    print("Loading...")
    s.load(dataset)
#    print("Evaluating baseline accuracy (all features)...")
#    s.evaluate_baseline()
    print("Running greedy, combined score...")
    s.use_accuracy_only = False
    s.greedy()
    if 0:
        print("Running greedy, accuracy only...")
        s.use_accuracy_only = True
        s.greedy()

###########################################

if __name__ == '__main__':
    main()
    print("all done!")

###########################################
