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

NUM_BINS = 256

###########################################
 
class State:
    def __init__(self):
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
    
    def eval_accuracy(self, indexes):
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

###########################################

def main():
    dataset = DEFAULT_DATASET
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    s = State()
    print("Loading...")
    s.load(dataset)
    print("Calculating mutual information...")
    r = s.mi()
    feature_indexes = []
    for i in range(MAX_FEATURES_GREEDY):
        score, name, index = r[i]
        feature_indexes.append(index)
        score, av, at, e = s.combined_score(feature_indexes)
        print("best at", name, score, av, at, e)

###########################################
    
if __name__ == '__main__':
    main()
    print("all done!")
