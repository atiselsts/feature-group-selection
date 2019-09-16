#!/usr/bin/python3 -u

#
# File: pso_algorithms.py
# Description: run the Particle Swarm Optimization based feature selection algorithms.
# Two versions are implemented:
#  - single-objective (select it by passing `s`) as a command line argument;
#  - multi-objective (the default).
# Author: Atis Elsts, 2018-2019
#

import os
import numpy as np
import random
import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import sys
sys.path.append("..")
sys.path.append("../energy-model")

import utils
import energy_model
from ml_config import *
import ml_state

###########################################

# the number of particles (30 in Xue's paper)
NUM_PARTICLES = 10000
# the number of iterations (100 in Xue's paper; usually converges faster?)
NUM_ITERATIONS = 100

# for testing
if False:
    NUM_PARTICLES = 10
    NUM_ITERATIONS = 2

INITIALIZE_WITH_ALL_PAIRS = True

# To make it prefer fewer particles, increase this
SELECTION_THRESHOLD = 0.9

# For initialization of the particle set
INITIAL_PROB = 0.5

# Settings from Xue's paper
W = 0.7298   # weight factor for inertia
C1 = 1.49618 # weight factor for the difference from personal best
C2 = 1.49618 # weight factor for the difference from global best
VMAX = 0.6   # 6 in the other one

# for multiobjective
NUM_DIMENSIONS = 2

###########################################

class Particle(object):
    def __init__(self, s, num_features):
        self.s = s
        self.num_features = num_features
        self.config = None

    def setup(self):
        if self.config:
            self.x = []
            for i in range(self.num_features):
                if i in self.config:
                    self.x.append(1)
                else:
                    self.x.append(0)
        else:
            # Initialize position in range 0..+1
            self.x = [1 if random.random() < INITIAL_PROB else 0 for _ in range(self.num_features)]
        # Initialize velocity in range -1..+1
        self.v = [2 * random.random() - 1.0 for _ in range(self.num_features)]
        # Initialize the particle's best known position to its initial position
        self.personal_best = copy.copy(self.x)
        # Initialize scores to nothing
        self.score = float("-inf")
        self.best_score = float("-inf")
        self.best_av = float("-inf")
        self.best_at = float("-inf")

    def move(self):
        # update position
        s = sum(1 for xi in self.x if xi >= SELECTION_THRESHOLD)
        for i in range(self.num_features):
            self.x[i] += self.v[i]
            if self.x[i] < 0.0:
                self.x[i] = 0.0
            elif self.x[i] > 1.0:
                self.x[i] = 1.0
        e = sum(1 for xi in self.x if xi >= SELECTION_THRESHOLD)
        #print("before:", s, "after", e)
        #print("d=", s - e)

        # update speed
        for i in range(self.num_features):
            d1 = self.personal_best[i] - self.x[i]
            d2 = self.s.best_particle.personal_best[i] - self.x[i]
            r1 = random.random()
            r2 = random.random()
            v_new = W * self.v[i] + C1 * r1 * d1 + C2 * r2 * d2
            #print(v_new, "=", "0.7 *", self.v[i], "+ 1.4 * ", r1, "*", d1 , "+ 1.4 * ", r2, "*", d2)
            self.v[i] = v_new
            if self.v[i] > VMAX:
                self.v[i] = VMAX
            elif self.v[i] < -VMAX:
                self.v[i] = -VMAX

    def get_indexes(self):
        result = []
        for i in range(self.num_features):
            if self.x[i] >= SELECTION_THRESHOLD:
                result.append(i)
        return tuple(result)

    def eval(self):
        indexes = self.get_indexes()
        self.score, av, at = self.s.score(indexes)
        # If the new position is better than best, take note of that
        if self.score > self.best_score:
            self.personal_best = copy.copy(self.x)
            self.best_score = self.score
            self.best_av = av
            self.best_at = at

    def __str__(self):
        indexes = self.get_indexes()
        names = sorted([self.s.groups[i] for i in indexes])

        av, at = self.s.eval_accuracy(indexes)
        e = self.s.eval_energy(indexes)

        return " Particle with #features={} accuracy={:.4f}/{:.4f} energy={:.4f} score={:.4f} features=[{}]".format(
            len(names), av, at, e, self.score, ",".join(names))

    def get_mscore():
        indexes = self.get_indexes()
        return (self.s.eval_accuracy(indexes), self.s.eval_energy(indexes))

###########################################

def copy_moparticle(p):
    r = MultiObjectiveParticle(p.s, p.num_features)
    r.x = copy.copy(p.x)
    r.v = copy.copy(p.v)
    r.personal_best = []
    r.personal_best.append(copy.copy(p.personal_best[0]))
    r.personal_best.append(copy.copy(p.personal_best[1]))
    r.score = copy.copy(p.score)
    r.best_score = copy.copy(p.best_score)
    r.best_av = p.best_av
    r.best_at = p.best_at
    return r

class MultiObjectiveParticle(Particle):
    def __init__(self, s, num_features):
        super(MultiObjectiveParticle, self).__init__(s, num_features)

    def setup(self):
        if self.config:
            self.x = []
            for i in range(self.num_features):
                if i in self.config:
                    self.x.append(1)
                else:
                    self.x.append(0)
        else:
            # Initialize position in range 0..+1
            self.x = [1 if random.random() < INITIAL_PROB else 0 for _ in range(self.num_features)]
        # Initialize velocity in range -1..+1
        self.v = [2 * random.random() - 1.0 for _ in range(self.num_features)]
        # Initialize the particle's best known position to its initial position
        self.personal_best = []
        self.personal_best.append(copy.copy(self.x))
        self.personal_best.append(copy.copy(self.x))
        # Initialize scores to nothing
        self.score = [float("-inf"), float("inf")]
        self.best_score = copy.copy(self.score)
        self.best_av = float("-inf")
        self.best_at = float("-inf")
        self.av = float("-inf")
        self.at = float("-inf")

    def eval(self):
        indexes = self.get_indexes()
        self.score, self.av, self.at = self.s.mscore(indexes)
        # convert back to nonscaled metrics
        a = self.score[0] / W_ACCURACY
        e = self.score[1] / W_ENERGY
        # bigger accuracy is better
        if a > self.best_score[0]:
            #print("update acc score: from ", self.best_score[0], "to", a)
            self.personal_best[0] = copy.copy(self.x)
            self.best_score[0] = a
            self.best_av = self.av
            self.best_at = self.at
        # smaller energy is better
        if e < self.best_score[1]:
            #print("update energy score: from ", self.best_score[1], "to", e)
            self.personal_best[1] = copy.copy(self.x)
            self.best_score[1] = e

    def __str__(self):
        indexes = self.get_indexes()
        names = sorted([self.s.groups[i] for i in indexes])
        e = self.score[1] / W_ENERGY
        return " Particle with #features={} accuracy={:.4f}/{:.4f} energy={:.4f} features=[{}]".format(
            len(names), self.av, self.at, e, ",".join(names))

    def __repr__(self):
        return str(self)

    # distance in result space
    def result_distance(self, other):
        d1 = self.score[0] - other.score[0]
        d2 = self.score[1] - other.score[1]
        # return squared distance: good enough for comparison
        return d1 * d1 + d2 * d2

    def move(self, all_gbest):
        # update position
        s = sum(1 for xi in self.x if xi >= SELECTION_THRESHOLD)
        for i in range(self.num_features):
            self.x[i] += self.v[i]
            if self.x[i] < 0.0:
                self.x[i] = 0.0
            elif self.x[i] > 1.0:
                self.x[i] = 1.0
        e = sum(1 for xi in self.x if xi >= SELECTION_THRESHOLD)
        #print("before:", s, "after", e)
        #print("d=", s - e)

        self.eval()

        # select one of the globally best particles to look up to
        if 0:
            gbest = random.choice(all_gbest)
        else:
            mindist = float("inf")
            gbest = None
            for g in all_gbest:
                d = self.result_distance(g)
                if d < mindist:
                    mindist = d
                    gbest = g

        # update speed 
        for i in range(self.num_features):
            d11 = self.personal_best[0][i] - self.x[i]
            d12 = self.personal_best[1][i] - self.x[i]
            if 0:
                # use best coordinates
                d21 = gbest.personal_best[0][i] - self.x[i]
                d22 = gbest.personal_best[1][i] - self.x[i]
            else:
                # use current coordinates
                d21 = gbest.x[i] - self.x[i]
                d22 = d21
            r1 = random.random()
            r2 = random.random()
            if 0:
                v_new = W * self.v[i] + C1 * r1 * (d11 + d12) + C2 * r2 * (d21 + d22)
            else:
                if random.random() > 0.5:
                    v_new = W * self.v[i] + C1 * r1 * d11 + C2 * r2 * d21
                else:
                    v_new = W * self.v[i] + C1 * r1 * d12 + C2 * r2 * d22

            #print(v_new, "=", "0.7 *", self.v[i], "+ 1.4 * ", r1, "*", d1 , "+ 1.4 * ", r2, "*", d2)
            self.v[i] = v_new
            if self.v[i] > VMAX:
                self.v[i] = VMAX
            elif self.v[i] < -VMAX:
                self.v[i] = -VMAX
        #print(["{:.2f}".format(x) for x in self.v])


###########################################

def make_particle(s, is_multi):
    if is_multi:
        return MultiObjectiveParticle(s, s.num_features)
    return Particle(s, s.num_features)

def make_particle_from_config(s, config, is_multi):
    if is_multi:
        p = MultiObjectiveParticle(s, s.num_features)
    else:
        p = Particle(s, s.num_features)
    p.config = list(config)
    return p

###########################################

class PSOState(ml_state.State):
    def __init__(self):
        super().__init__()
        # global best of the swarm
        self.best_particle = None
        # already evaluated positions
        self.cache = {}
        # already evaluated positions: vector of scores for multi-objective optimization
        self.mcache = {}

    def init_particles(self, is_multi):
        self.particles = []

        if INITIALIZE_WITH_ALL_PAIRS:
            # initialize with all possible pairs of particles
            for i in range(self.num_features):
                for j in range(i + 1, self.num_features):
                    indexes = (i, j)
                    self.particles.append(make_particle_from_config(self, indexes, is_multi))
                    if len(self.particles) >= NUM_PARTICLES:
                        break
                if len(self.particles) >= NUM_PARTICLES:
                        break
            #print("num particles=", len(self.particles))

        # Initialize with extra, random particles
        while len(self.particles) < NUM_PARTICLES:
            self.particles.append(make_particle(self, is_multi))

        # Get initial score
        for p in self.particles:
            p.setup()
            p.eval()
        # Initialize the new global best
        self.best_particle = self.particles[0]
        for p in self.particles[1:]:
            if p.score > self.best_particle.best_score:
                self.best_particle = p

    def score(self, indexes):
        # this was already seen?
        if indexes not in self.cache:
            av, at = self.eval_accuracy(indexes)
            e = self.eval_energy(indexes)
            score = roundacc(W_ACCURACY * av) + W_ENERGY * e
            self.cache[indexes] = (score, av, at)
        return self.cache[indexes]

    def mscore(self, indexes):
        # this was already seen?
        if indexes not in self.mcache:
            av, at = self.eval_accuracy(indexes)
            e = self.eval_energy(indexes)
            score = [roundacc(W_ACCURACY * av), W_ENERGY * e]
            self.mcache[indexes] = (score, av, at)
        return self.mcache[indexes]


# Sorting functions
def sort(s):
    s.sort(key = lambda p: p.score, reverse=True)

def nondominated_sort(s):
    # sort by accuracy first (higher accuracy comes first)
    # then energy (lower energy comes first)
    s.sort(key = lambda p: p.score, reverse=True)
    # this is the Pareto front
    f1 = [s[0]]
    rest = []
    for i, candidate in enumerate(s[1:]):
        in_front = True
        # iterate for all particles that come before this one
        for dom_part in f1:
            # if this one has better energy, drop the candidate
            if dom_part.score[1] >= candidate.score[1]:
                in_front = False
                break
        if in_front:
            f1.append(candidate)
        else:
            rest.append(candidate)
    return f1, rest


def sort_by_crowding(s):
    for p in s:
        p.crowding_distance = [0] * NUM_DIMENSIONS

    # `s` is assumed to be already sorted by score,
    # and, given that this is a Pareto front,
    # if means that they will be sorted in all dimensions in the same time
    for dimension in range(NUM_DIMENSIONS):
        # calculate the crowding in this dimension
        for i in range(len(s)):
            if i == 0 or i == len(s) - 1:
                s[i].crowding_distance[dimension] = float("inf")
            else:
                d1 = s[i].score[dimension] - s[i - 1].score[dimension]
                d2 = s[i + 1].score[dimension] - s[i].score[dimension]
                if dimension == 0:
                    d1 = -d1
                    d2 = -d2
                assert d1 >= 0
                assert d2 >= 0
                s[i].crowding_distance[dimension] = d1 + d2
    # the ones with higher crowding distance are better (less crowded)
    return sorted(s, key = lambda p: sum(p.crowding_distance), reverse=True)

###########################################

#
# Single-objective particle swarm optimization
#
def so_pso(dataset):
    print("Single objective")
    s = PSOState()
    print("Loading...")
    s.load(dataset)
    print("Initializing starting positions and scores...")
    s.init_particles(False)

    print("Initialization done, initial Pareto front:")
    sort(s.particles)
    for p in s.particles[:10]:
        print(" ", p)

    for it in range(NUM_ITERATIONS):
        print("Iteration", it)
        # Move to a new position
        for p in s.particles:
            p.move()
        # Evaluate in the new position
        for p in s.particles:
            p.eval()
        for p in s.particles:
            # check if there's a new global best
            if p.score > s.best_particle.best_score:
                # update the global best
                #print("new gb")
                s.best_particle = p
        print("Best: {}".format(s.best_particle))

    sort(s.particles)
    for p in s.particles:
        print(p)

    print("\nFinal Pareto front")
    # treat as multidimensional optimization and find the Pareto front
    mp = []
    seen_indexes = set()
    for p in s.particles:
        mp.append(MultiObjectiveParticle(s, s.num_features))
        mp[-1].setup()
        indexes = mp[-1].get_indexes()
        if indexes in seen_indexes:
            continue # already have this particle
        seen_indexes.add(indexes)
        mp[-1].x = p.x
        mp[-1].eval()

    f1, _ = nondominated_sort(mp)
    for p in f1:
        print(" ", p)
    print("")

###########################################

#
# Multi-objective particle swarm optimization based on nondominant sorting ideas
#
def mo_pso(dataset):
    print("Multi objective")
    s = PSOState()
    print("Loading...")
    s.load(dataset)
    print("Initializing starting positions and scores...")
    s.init_particles(True)

    print("Initialization done, initial Pareto front:")
    f1, rest = nondominated_sort(s.particles)
    for p in f1:
        print(" ", p)

    for it in range(NUM_ITERATIONS):
        print("Iteration", it)

        f1, _ = nondominated_sort(s.particles)
        f1 = sort_by_crowding(f1)
        # take half of F1 as the "highest ranked (least crowded) solutions in nonDomS" from the paper
        num_to_take = (3 * len(f1) + 3) // 4
        highest_ranked_f1 = f1[:num_to_take]

        union = []
        for p in s.particles:
            # Insert the particle with the old position and score
            union.append(p)
            # Create a new particle based on the old one
            p1 = copy_moparticle(p)
            # Move to a new position
            p1.move(highest_ranked_f1)
            # Insert the particle with the new position
            union.append(p1)

        # start afresh
        s.particles = []
        while len(s.particles) < NUM_PARTICLES:
            f1, rest = nondominated_sort(union)
            if len(f1) + len(s.particles) <= NUM_PARTICLES:
                # fits fully
                s.particles += f1
            else:
                # fits only partially
                f1 = sort_by_crowding(f1)
                i = 0
                while len(s.particles) < NUM_PARTICLES:
                    s.particles.append(f1[i])
                break
            # remove the F1 front from the union
            union = [p for p in rest]

    print("Final Pareto front:")
    f1, _ = nondominated_sort(s.particles)
    for p in f1:
        print(" ", p)

###########################################

def main():
    print("Num particles =", NUM_PARTICLES)
    dataset = DEFAULT_DATASET
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == "s":
        do_single = True
    else:
        do_single = False

    if do_single:
        so_pso(dataset)
    else:
        mo_pso(dataset)

def har_multi():
    for i in range(10):
        mo_pso("UCI HAR Dataset")

###########################################

if __name__ == '__main__':
    for n in [10, 30, 100, 300, 1000, 3000]:
        NUM_PARTICLES = n
        main()
        print("")

#    if True:
#      main()
#    else:
#      har_multi()
#    print("all done!")
