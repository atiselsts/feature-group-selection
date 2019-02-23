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

###########################################
   
class State:
    def __init__(self):
        # whether to use accuracy only of the combined energy accuracy score
        self.use_accuracy_only = False
        # whether to operate at group or individual vector level
        self.do_subselection = False

    def load(self, dataset):
        filename = os.path.join("..", "datasets", dataset, "train", "features.csv")
        self.train = np.asarray(utils.load_csv(filename, skiprows=1))
        filename = os.path.join("..", "datasets", dataset, "train", "y_train.txt")
        self.train_y = np.asarray(utils.load_csv(filename)).ravel()

        filename = os.path.join("..", "datasets", dataset, "validation", "features.csv")
        self.validation = np.asarray(utils.load_csv(filename, skiprows=1))
        filename = os.path.join("..", "datasets", dataset, "validation", "y_validation.txt")
        self.validation_y = np.asarray(utils.load_csv(filename)).ravel()

        filename = os.path.join("..", "datasets", dataset, "test", "features.csv")
        self.test = np.asarray(utils.load_csv(filename, skiprows=1))
        filename = os.path.join("..", "datasets", dataset, "test", "y_test.txt")
        self.test_y = np.asarray(utils.load_csv(filename)).ravel()

        if USE_N_FOLD_CROSS_VALIDATION:
            self.alltrain = np.concatenate((self.train, self.validation, self.test))
            self.alltrain_y = np.concatenate((self.train_y, self.validation_y, self.test_y))

        filename = os.path.join("..", "feature_names.csv")
        self.names = utils.read_list_of_features(filename)

        if self.do_subselection:
            self.groups = [n[1] for n in self.names]
        else:
            # need to preserve order, so cannot uniquify via the usual way (via a set)
            self.groups = []
            for n in self.names:
                if n[2] not in self.groups:
                    self.groups.append(n[2])

        self.num_features = len(self.groups) # number of features

    def evaluate_baseline(self):
        validation_scores = []
        test_scores = []
        for i in range(10):
            clf = RandomForestClassifier(n_estimators = NUM_TREES, random_state=i,
                                         class_weight = "balanced")

            clf.fit(self.train, self.train_y)

            hypothesis = clf.predict(self.validation)
            f1 = f1_score(self.validation_y, hypothesis, average="micro")
            validation_scores.append(f1)

            hypothesis = clf.predict(self.test)
            f1 = f1_score(self.test_y, hypothesis, average="micro")
            test_scores.append(f1)
        s_test = np.mean(test_scores)
        s_validation = np.mean(validation_scores)
        validation_scores = ["{:.4f}".format(x) for x in validation_scores]
        test_scores = ["{:.4f}".format(x) for x in test_scores]
        print("validation:" , "{:.4f}".format(s_validation), validation_scores)
        print("test      :" , "{:.4f}".format(s_test), test_scores)

    def eval_accuracy(self, indexes):
        if len(indexes) == 0:
            return RANDOM_ACCURACY
        selector = utils.select(self.names, self.groups, indexes, self.do_subselection)

        if USE_N_FOLD_CROSS_VALIDATION:
            features_alltrain = self.alltrain[:,selector]
            validation_score = 0
            rs = ShuffleSplit(n_splits = NUM_VALIDATION_ITERATIONS, test_size = 0.33)
            scores = []
            # use balanced weigths to account for class imbalance
            # (we're trying to optimize f1 score, not accuracy?)
            clf = RandomForestClassifier(n_estimators = NUM_TREES, random_state=0,
                                         class_weight = "balanced")
            for train_index, test_index in rs.split(features_alltrain):
                clf.fit(features_alltrain[train_index], self.alltrain_y[train_index])
                s = clf.score(features_alltrain[test_index], self.alltrain_y[test_index])
                scores.append("{:2.2f}".format(s))
                validation_score += s
            validation_score /= NUM_VALIDATION_ITERATIONS
            test_score = validation_score
        else:
            # simply train and then evaluate
            features_train = self.train[:,selector]
            features_validation = self.validation[:,selector]
            scores = []
            for i in range(NUM_TRIALS):
                # use balanced weigths to account for class imbalance
                # (we're trying to optimize f1 score, not accuracy?)
                clf = RandomForestClassifier(n_estimators = NUM_TREES, random_state=i,
                                             class_weight = "balanced")
                clf.fit(features_train, self.train_y)
                #validation_score = clf.score(features_validation, self.validation_y)
                hypothesis = clf.predict(features_validation)
                #c = (self.validation_y == hypothesis)
                f1 = f1_score(self.validation_y, hypothesis, average="micro")
                scores.append(f1)
            validation_score = np.mean(scores)

            # check also the results on the test set
            features_test = self.test[:,selector]
            hypothesis = clf.predict(features_test)
            f1 = f1_score(self.test_y, hypothesis, average="micro")
            test_score = f1

        #print("validation={:.2f} test={:.2f}".format(validation_score, test_score))
        return validation_score, test_score

    def eval_energy(self, indexes):
        names = [self.groups[i] for i in indexes]
        #print("names=", names)
        return sum(energy_model.calc(names))

    def combined_score(self, indexes):
        av, at = self.eval_accuracy(indexes)
        b = self.eval_energy(indexes)
        score = roundacc(W_ACCURACY * av) + W_ENERGY * b
        return score, av, at, b

    def greedy(self):
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

        print("best at", self.groups[best_f], best_score, best_av, best_at, best_e)
        if len(used_features) + 1 < MAX_FEATURES_GREEDY or best_score >= prev_best - 0.001:
            updated_features = copy.copy(used_features)
            updated_features.append(best_f)
            print("one level deeper, used=", [self.groups[x] for x in updated_features])
            self.greedy_iteration(updated_features, best_score)
        else:
            print("stopping: not better than previous: {:.4f} vs {:.4f}".format(
                best_score, prev_best))

###########################################

def main():
    dataset = DEFAULT_DATASET
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    s = State()
    print("Loading...")
    s.load(dataset)
    print("Evaluating baseline accuracy (all features)...")
    s.evaluate_baseline()
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
