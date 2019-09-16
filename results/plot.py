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

def plot(filename, labels, data, use_test_scores, xlim=None):
    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))

    labels = labels[::-1]
    data = data[::-1]

    for l, d in zip(labels, data):
        x = [u[0] for u in d]  # energy
        if use_test_scores:
            y = [u[2] for u in d] # test
        else:
            y = [u[1] for u in d] # validation
        ax.plot(x, y, label=l[0], marker=l[2], markersize=6)

#    ax.set_ylim(0.5, 0.92)
    ax.set_ylim(0.41, 1.0)
    if xlim is None:
        xlim = 35
    ax.set_xlim(0, xlim)
    if SHOW_RAW:
        pl.gca().axvline(costs_tx["raw"], 0, 200, color="red")

    # pl.title("HAR dataset (6 classes)")
    ax.set_ylabel("Classification F1 score")
    ax.set_xlabel("Charge, microcoulumbs per window (128 samples)")

#    legend = pl.legend(#bbox_to_anchor=(0.5, 1.1), # loc='upper center', ncol=4,
#                       prop={'size':10},
#                       handler_map={lh.Line2D: lh.HandlerLine2D(numpoints=1)})

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], labels[::-1], prop={'size': 9})
#              title='Line', loc='upper left')

    pl.savefig(filename,
               format='pdf',
               bbox_extra_artists=(legend,),
               bbox_inches='tight')
    pl.close()

###########################################

def plot_fn(filename, data):
    fig, ax = pl.subplots()
    fig.set_size_inches((4, 2.5))

    c = Counter()
    c_pairs = Counter()

    all_feature_groups = []

    total_groups = 0
    print("plot", filename)
    for d in data:
        for group in d:
            if len(group) == 0:
                continue
            any_nonempty_feature = False
            fixed_names_group = []
            for feature in group:
                f = feature.replace("tTotalAcc", "").lstrip("-")
                if f == "":
                    continue
                fixed_names_group.append(f)
                any_nonempty_feature = True
                c[f] += 1 # increase the count

            if not any_nonempty_feature:
                continue

            all_feature_groups.append(fixed_names_group)

            total_groups += 1
            for feature1 in group:
                for feature2 in group:
                    if feature1 == feature2:
                        continue
                    key = sorted([feature1, feature2])
                    c_pairs[(key[0], key[1])] += 1 # increase the count
        all_feature_groups.append(["---"])

    features = []
    values = []
    for name, occurences in c.most_common()[:10]:
        features.append(name)
        values.append(100.0 * occurences / total_groups)

    feature_pairs = []
    values_pairs = []
    for names, occurences in c_pairs.most_common()[:10]:
        feature_pairs.append(names)
        values_pairs.append(100.0 * occurences / total_groups)

    print("pairs")
    for p, v in zip(feature_pairs, values_pairs):
        print(p, v)
    print("")

    y_pos = np.arange(len(values))
    pl.xlim(0, 80)

    ax.barh(y_pos, values, align='center',
            color='brown', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Occurence probability in a Pareto-front group, %')

    pl.savefig(filename,
               format='pdf',
#               bbox_extra_artists=(legend,),
               bbox_inches='tight')
    pl.close()

    # dump the features to a file
    filename = filename.replace(".pdf", ".txt")
    with open(filename, "w") as f:
        for gr in all_feature_groups:
            f.write(",".join(gr) + "\n")


###########################################

def parse_pso(dataset, suffix):
    filename = dataset + "_" + suffix + ".log"
    if not os.access(filename, os.R_OK):
        filename = filename.lower()
        if not os.access(filename, os.R_OK):
            print("no such file:", filename)
            return []

    with open(filename, "r") as f:
        lines = [x.strip() for x in f.readlines()]

    result = []
    feature_names = []
    multi_result = []
    multi_feature_names = []
    parsing = False
    for line in lines:
        if parsing == False:
            if "Final Pareto front" not in line:
                continue
            parsing = True
        else:
            if "Multi objective" in line:
                multi_result.append(result)
                multi_feature_names.append(feature_names)
                result = []
                feature_names = []
                parsing = False
            #Particle with #features=2 accuracy=0.5395/0.5908 energy=1.9887 features=[tTotalAccMagSq-iqr(),tTotalAccMagSq-median()]
            elif "Particle with" in line:
                fields = line.split()
                acc = fields[3]
                av, at = acc.split("=")[1].split("/")
                acc_validation = float(av)
                acc_test = float(at)
                energy = float(fields[4].split("=")[1])
                features = fields[5].split("=")[1].replace("(", "").replace(")", "")
                features = features[:-1][1:]
                features = features.split(",")
                feature_names.append(features)
                result.append((energy, acc_validation, acc_test))
    if len(multi_result):
        if len(result):
            multi_result.append(result)
            multi_feature_names.append(feature_names)
        return multi_result, multi_feature_names
    return result, feature_names

###########################################

def parse_greedy(dataset, suffix):
    filename = dataset + "_" + suffix + ".log"
    if not os.access(filename, os.R_OK):
        filename = filename.lower()
        if not os.access(filename, os.R_OK):
            print("no such file:", filename)
            return []
    
    with open(filename, "r") as f:
        lines = [x.strip() for x in f.readlines()]

    result = []
    feature_names = []

    for l in lines:
        if "best at" not in l:
            continue
        #best at tTotalAcc-correlation() 410.911019 0.83784800602 0.823549372243 8.088981
        fields = l.split()
        name = fields[2].replace("(", "").replace(")", "")
        acc_validation = float(fields[4])
        acc_test = float(fields[5])
        energy = float(fields[6])

        if len(feature_names):
            feature_names.append(feature_names[-1] + [name])
        else:
            feature_names.append([name])

        if len(result):
            old_energy, old_acc_validation, old_acc_test = result[-1]
            if energy == old_energy and acc_validation == old_acc_validation and acc_test == old_acc_test:
                # do not add duplicates
                continue
        result.append((energy, acc_validation, acc_test))

    #print("r=", result)
    return result, feature_names

###########################################

def viz_generation_selection(name, suffix, colour):
    """ Code from NT to plot the selection and cost of features over the generations. 
        colour (in {Val, Test, Energy}) specifies the variable with which the 
        heatmap is coloured """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as pl 
    
    evt, fn = dict(
        m=parse_pso, 
        s=parse_pso, 
        greedy=parse_greedy, 
        mi=parse_greedy, 
    )[suffix](name, suffix)
    
    evt = pd.DataFrame(evt)
    evt.columns = 'Energy', 'Val', 'Test'
    
    k = 'Energy'
    
    d = []
    for fi, (ri, rr) in zip(fn, evt.iterrows()): 
        di = {kk: rr[colour] for kk in fi}
        di[k] = rr[k]
        d.append(di)
    fn = pd.DataFrame(d).fillna(0).set_index(k).sort_index().reset_index()
    del fn[k]

    inds = fn.ne(0).idxmax()
    col_ord = [fn.columns[ii] for ii in np.argsort(inds.values)]
    
    ll = 7.5
    fig, ax = pl.subplots(1, 1, figsize=(ll * 2, ll))
    sns.heatmap(fn[col_ord].T, cmap='RdBu_r') 
#    pl.title(f'{name}_{suffix}')
    pl.tight_layout()
#    fn = f'nt/{name}_{suffix}_{colour.lower()}.pdf'
    pl.savefig(fn)
    pl.close()
    return fn

###########################################

def read_data(dataset):
    data = []
    feature_names = []
    for c in CATEGORIES:
        d, fn = c[3](dataset, c[1])
        data.append(d)
        if c[1] == "mi": # skip for mutual information, as that is not very interesting
            fn = []
        feature_names.append(fn)
    return data, feature_names

###########################################

def main():
    suffix = "m_multi"
    data, fn = parse_pso("har", suffix)
    labels = [("Run " + str(x), suffix, "o") for x in range(1, 11)]
    plot("har_multi_validation.pdf", labels, data, False, xlim=28)
    plot("har_multi_test.pdf", labels, data, True, xlim=28)
    return

    feature_names_per_algorithm = [[] for _ in CATEGORIES]
    for dataset in utils.ALL_DATASETS_SHORT:
        data, feature_names = read_data(dataset)
        plot(dataset + "_validation.pdf", CATEGORIES, data, False)
        plot(dataset + "_test.pdf", CATEGORIES, data, True)
        plot_fn("fn_by_ds_" + dataset + ".pdf", feature_names)

        for i, c in enumerate(CATEGORIES):
            feature_names_per_algorithm[i].append(feature_names[i])

    har_data = {}
    
    har_data[10000], _ = parse_pso("har", "m")
    for numparticles in [10, 30, 100, 300, 1000, 3000]:
        har_data[numparticles], _ = parse_pso("har", "num_particles_" + str(numparticles))

    keys = sorted(list(har_data.keys()))[:-1]
    for numparticles in keys:
        data = [
            har_data[numparticles],
            har_data[10000],
        ]
        labels = [
            (str(numparticles) + " particles", "", "o"),
            ("10000 particles", "", "o")
        ]
        plot("har_{}_particles_validation.pdf".format(numparticles), labels, data, False, xlim=28)
        plot("har_{}_particles_test.pdf".format(numparticles), labels, data, True, xlim=28)


#    feature_names_all = []
#    for i, c in enumerate(CATEGORIES):
#        plot_fn("fn_by_algorithm_" + c[1] + ".pdf", feature_names_per_algorithm[i])
#        for gr in feature_names_per_algorithm[i]:
#            feature_names_all.append(gr)

#    plot_fn("fn_by_all.pdf", feature_names_all)

###########################################
    
if __name__ == '__main__':
    CATEGORIES.append(("Multiobjective PSO", "m", "o", parse_pso))
    CATEGORIES.append(("Single objective PSO", "s", "s", parse_pso))
    CATEGORIES.append(("Greedy search", "greedy", "d", parse_greedy))
    CATEGORIES.append(("Mutual information", "mi", "^", parse_greedy))
    main()
    print("all done!")
