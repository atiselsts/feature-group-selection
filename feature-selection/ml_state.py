import os
import numpy as np
import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import f1_score

import sys
sys.path.append("..")
sys.path.append("../energy-model")

import utils
import energy_model
from ml_config import *

class State:
    def __init__(self):
        # whether to use accuracy only of the combined energy accuracy score
        self.use_accuracy_only = False
        # whether to operate at group or individual vector level
        self.do_subselection = False

    def load_subset(self, dataset, name):
        filename = os.path.join("..", "datasets", dataset, name, "features.csv")
        data = np.asarray(utils.load_csv(filename, skiprows=1))
        filename = os.path.join("..", "datasets", dataset, name, "y_{}.txt".format(name))
        activities = np.asarray(utils.load_csv(filename)).ravel()
        filename = os.path.join("..", "datasets", dataset, name, "subject_{}.txt".format(name))
        subjects = np.asarray(utils.load_csv(filename)).ravel()
        return data, activities, subjects

    def load(self, dataset):
        self.train, self.train_y, self.train_subjects = self.load_subset(dataset, "train")
        self.validation, self.validation_y, self.validation_subjects = self.load_subset(dataset, "validation")
        self.test, self.test_y, self.test_subjects = self.load_subset(dataset, "test")

        if USE_N_FOLD_CROSS_VALIDATION:
            self.alltrain = np.concatenate((self.train, self.validation, self.test))
            self.alltrain_y = np.concatenate((self.train_y, self.validation_y, self.test_y))
            self.alltrain_subjects = np.concatenate((self.train_subjects, self.validation_subjects, self.test_subjects))
            # just pick the first one
            self.subject_left_out = self.alltrain_subjects[0]

            self.cv = []
            self.cv_y = []

            self.left_out = []
            self.left_out_y = []

            for i in range(len(self.alltrain_subjects)):
                subject = self.alltrain_subjects[i]
                if subject == self.subject_left_out:
                    self.left_out.append(self.alltrain[i])
                    self.left_out_y.append(self.alltrain_y[i])
                else:
                    self.cv.append(self.alltrain[i])
                    self.cv_y.append(self.alltrain_y[i])

            print("number of the subject left out:", int(self.subject_left_out))

            self.cv = np.asarray(self.cv)
            self.cv_y = np.asarray(self.cv_y).ravel()

            self.left_out = np.asarray(self.left_out)
            self.left_out_y = np.asarray(self.left_out_y).ravel()


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

        # get the energy for raw data, used to stop iterating
        self.energy_for_raw = self.eval_energy_for_raw()
        print("Stopping energy value is {:.4f}".format(self.energy_for_raw))

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
            return RANDOM_ACCURACY, RANDOM_ACCURACY

        selector = utils.select(self.names, self.groups, indexes, self.do_subselection)

        if USE_N_FOLD_CROSS_VALIDATION:
            features = self.cv[:,selector]
            left_out_features = self.left_out[:,selector]
            validation_score = 0
            test_score = 0
#            rs = ShuffleSplit(n_splits = NUM_VALIDATION_ITERATIONS, test_size = 0.33)
            rs = KFold(n_splits = NUM_VALIDATION_ITERATIONS) #, test_size = 0.33)
            scores = []
            # use balanced weigths to account for class imbalance
            # (we're trying to optimize f1 score, not accuracy)
            clf = RandomForestClassifier(n_estimators = NUM_TREES, random_state=0,
                                         class_weight = "balanced")
            for train_index, test_index in rs.split(features):
                clf.fit(features[train_index], self.cv_y[train_index])
#                s1 = clf.score(features[test_index], self.cv_y[test_index])
#                s2 = clf.score(left_out_features, self.left_out_y)

                hypothesis = clf.predict(features[test_index])
                s1 = f1_score(self.cv_y[test_index], hypothesis, average="micro")

                hypothesis = clf.predict(left_out_features)
                s2 = f1_score(self.left_out_y, hypothesis, average="micro")

                scores.append("{:2.2f} ({:2.2f})".format(s1, s2))
                validation_score += s1
                test_score += s2
            validation_score /= NUM_VALIDATION_ITERATIONS
            test_score /= NUM_VALIDATION_ITERATIONS
#            names = [self.groups[i] for i in indexes]
#            print(names)
#            print("validation/test:" , scores)
        else:
            # simply train and then evaluate
            features_train = self.train[:,selector]
            features_validation = self.validation[:,selector]
            scores = []
            for i in range(NUM_TRIALS):
                # use balanced weigths to account for class imbalance
                # (we're trying to optimize f1 score, not accuracy)
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

    def eval_energy_for_raw(self):
        return sum(energy_model.calc_raw())

