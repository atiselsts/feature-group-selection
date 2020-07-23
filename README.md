This repository contains code for feature selection for human activity recognition applications using acceleremoter data. The feature selection is done taking into account both accuracy (F1 score) and energy efficiency, making the features suitable for low-energy IoT devices. 


This repository has the code and datasets referenced in the paper *Atis Elsts, Niall Twomey, Ryan McConville, and Ian Craddock. Energy-efficient activity recognition framework using wearable accelerometers. Elsevier Journal of Computer Networks and Applications, In Press. https://doi.org/10.1016/j.jnca.2020.102770*.

## Structure ##

The structure of this repository:

* `datasets` - raw acceleration data for each of the datasets used. The data is raw, but already cleaned up.
* `datasets/HAR`- this is the HAR dataset data. It has 6 activities.
* `datasets/SPHERE/open_sphere_challenge_data` - the is the data from the web. Only the six best participants are selected (in terms of packet delivery rate) and can be found there, since the others had too many missing data. The files `acceleration_corrected.csv` have the data that has already rotated.
* `datasets/PAMAP2` - this is the PAMAP2 dataset. It has 18 activities in total, 12 "protocol" activities.
* `feature-selection` - feature selection algorithms.
* `c-implementation` - the C feature extraction library that can be compiled to run on embedded devices. This library is used to construct our empirical energy model.
* `results` - our existing results, published in the paper: figures and log files.

## Getting started ##

To prepare the data for running the feature selection algorithms, run these steps (they are explained below):

```
$ pip install -r requirements.txt
$ cd datasets
$ cd SPHERE; ./preprocess_and_cleanup.py; cd ..
$ cd "UCI HAR Dataset"; ./preprocess_and_cleanup.py; cd ..
$ # this is not needed since the full PAMAP2 datset is not included due to its size
$ # cd PAMAP2; ./preprocess_and_cleanup.py; cd ..
$ # this will do the job for all datasets
$ ./extract-features.py
```

To run a feature selection algorithms, e.g. the greedy selection:

```
$ cd feature-selection
$ ./greedy_algorithms.py SPHERE
```

If updating the embedded feature extraction and/or the energy model:
* Compile the C code in `c-implementation`, upload it to a device, measure the time needed to calculate the features
* Run the `./output` executable in the `c-implementation` folder to obtain on-board output of the feature estimation
* Update the `.py` files under `energy-model.py/` with the new values (see the files for more info).

Normally you don't need to do this - just use the existing file  `energy-model.py/energy-model.py`. It has an energy model that is already prepared for use.


## Details ##

### Data preprocessing and cleanup ###

Data is preprocessed to prepare it in a format for feature extraction.

The output format of this step is a list of raw acceleration data that matches the format of the HAR dataset. Namely:

* "Holes" (i.e. missing samples) in the data are plugged in, by filling repeated copies of the last measurement before the hole. The % of data missing is small (a few % max), so no big impact on the evaluation is expected. 
* In the `Inertial Signals` subfolder, an accelerometer raw data file for each of the axis is created.
* The format of the file is 128 records per a line. The records are 50% overlapping with the previous window, so that file is 50% redundant.
* The files `y_train.txt` / `y_validation.txt` / `y_test.txt`  contain the 2/3 majority label for each window. (The prefix `y` comes from the fact that it contains the ground truth, i.e. answers to the `X \theta = y` equation, not form `y` axis of the acceleration.)
* The data that does not have a clear 2/3 majority within a windows is marked with activity -1.

Then the  data is separated in train (~50%), validation (~25%), and test (~25%) sets. Note that cross validation is not used due to speed issues - some of the feature selection algorithms require high computation load.

What you need to do:
* Go to `datasets/SPHERE`, run the file `preprocess_and_cleanup.py`.
* Go to `datasets/UCI HAR Dataset`, run the file `separate_validation.py
* For the PAMAP2 dataset, the preprocessed data is already included in this repository, as the original data is too big. The preprocessing script `extract_and_split.py` is nevertheless provided in the dataset's directory.

The expected outcome is that there are `train`, `validation`, and `test` folders in each dataset's directory with the raw, segmented input data, and also activity labels.

### Feature extraction ###

Subsequently, features are extracted from the raw data.

That is done by running the script `extract-features.py` in the `datasets` directory, passing the dataset name as a parameter.

The expected outcome is that there are files `features.csv` in each of the `train`, `validation`, and `test` folders in each dataset's directory.

The file `feature_names.csv` in the top directory also is going to be rewritten / created if it does not exists. It shows the list of all features.

### Feature selection ###

These algorithms are supported:

* Greedy search algorithm
* Particle Swarm Optimization (PSO)
* Mutual information

Greedy search algorithm: selects the list of the features one by one, until the list is long enough AND no significant progress is obtained. Two objective functions are used: 1) accuracy-based; 2) using a single metric that combines energy and accuracy in a weighted way.

Particle Swarm Optimization has two implementations:

* PSO single objective algorithm: optimizes the Pareto front of feature groups by using a single metric that combines energy and accuracy in a weighted way.
* PSO multi objective: optimizes the Pareto front of feature groups by using two different metrics: accuracy and energy, independently.

The "mutual information" method calculates the [mutual information](https://en.wikipedia.org/wiki/Mutual_information) between each feature and the labels. After that, a list of features can be selected in a greedy fashion.
