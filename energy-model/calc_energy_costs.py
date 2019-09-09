#!/usr/bin/python3

import cbor
import os
import sys

sys.path.append("..")
import utils

#
# This file calculates energy costs for the CPU and radio.
# Note that the results are for single axis only!
#

# Inputs:
#  1) The time it takes to calculate the features on a CC2650 device
#     This should be copied directly into the code, in the variable named "results".
#  2) The data produced by the feature extraction.
#     This should be in a file ./out.txt. Assuming `make` has been run in the
#     `c-implementation` folder, the file can be produced by:
#     $ ../c-implementation/output-test > out.txt
#

DO_PRINT_AVG_CURRENT = False
SAMPLING_RATE = 50

########################################

WINDOW_SIZE_SAMPLES = utils.WINDOW_SIZE_SAMPLES

MAX_LEN = 64

CURRENTS_MA = {
    "SPW-2" : {
        "cpu" : 2.703,
        "rx" : 7.67,  # XXX
        "tx" : 12.0,  # guesswork based on the rocketlogger graphs!
        "sleep" : 1.335,
        "deep-sleep" : 0.016 # 16 uA average
    },
}

CURRENT_CPU = CURRENTS_MA["SPW-2"]["cpu"]
CURRENT_TX  = CURRENTS_MA["SPW-2"]["tx"]

TX_EFFICIENCY = 0.5   # assume that the packet has 50% overhead
BYTE_TIME_USEC = 32.0 # using 802.15.4

###################################################################

def get_processing_charge_uc_per_time(activity_duration_usec):
    return activity_duration_usec * CURRENT_CPU / 1000.0

def get_tx_charge_uc_per_bytes(num_bytes):
    tx_time_usec = num_bytes * BYTE_TIME_USEC / TX_EFFICIENCY
    return tx_time_usec * CURRENT_TX / 1000.0

###################################################################


old_results = """
Feature: nop Time: 3203 usec per 15000 samples
Feature: nop_nop Time: 3853 usec per 15000 samples
Feature: mean Time: 6640 usec per 15000 samples
Feature: energy Time: 6666 usec per 15000 samples
Feature: energy+mean Time: 6770 usec per 15000 samples
Feature: std Time: 9193 usec per 15000 samples
Feature: std+mean Time: 9216 usec per 15000 samples
Feature: std+energy Time: 9846 usec per 15000 samples
Feature: std+energy+mean Time: 8436 usec per 15000 samples
Feature: correlation Time: 17290 usec per 15000 samples
Feature: correlation+std Time: 17266 usec per 15000 samples
Feature: correlation+std+std Time: 17423 usec per 15000 samples
Feature: entropy Time: 66820 usec per 15000 samples
Feature: min Time: 6643 usec per 15000 samples
Feature: min+max Time: 7916 usec per 15000 samples
Feature: median Time: 16770 usec per 15000 samples
Feature: iqr Time: 18226 usec per 15000 samples
Feature: median+iqr Time: 21170 usec per 15000 samples
Feature: median+iqr+min+max Time: 23776 usec per 15000 samples
Feature: sma Time: 7916 usec per 15000 samples
Feature: t_median Time: 8673 usec per 15000 samples
Feature: t_jerk Time: 6016 usec per 15000 samples
Feature: t_magnitude_sq Time: 7630 usec per 15000 samples
Feature: t_magnitude Time: 123570 usec per 15000 samples
Feature: t_jerk+magnitude_sq Time: 12473 usec per 15000 samples
Feature: t_jerk+magnitude_f Time: 89010 usec per 15000 samples
"""

results = """
Feature: nop Time: 3226 usec per 15000 samples
Feature: nop_nop Time: 3830 usec per 15000 samples
Feature: mean Time: 6770 usec per 15000 samples
Feature: energy Time: 8436 usec per 15000 samples
Feature: energy+mean Time: 9246 usec per 15000 samples
Feature: std Time: 9190 usec per 15000 samples
Feature: std+mean Time: 9220 usec per 15000 samples
Feature: std+energy Time: 10260 usec per 15000 samples
Feature: std+energy+mean Time: 10363 usec per 15000 samples
Feature: correlation Time: 17320 usec per 15000 samples
Feature: correlation+std Time: 17290 usec per 15000 samples
Feature: correlation+std+std Time: 17320 usec per 15000 samples
Feature: entropy Time: 66926 usec per 15000 samples
Feature: min Time: 6770 usec per 15000 samples
Feature: min+max Time: 8046 usec per 15000 samples
Feature: median Time: 16770 usec per 15000 samples
Feature: iqr Time: 18203 usec per 15000 samples
Feature: median+iqr Time: 21196 usec per 15000 samples
Feature: median+iqr+min+max Time: 23803 usec per 15000 samples
Feature: t_median Time: 8696 usec per 15000 samples
Feature: t_l1norm Time: 8906 usec per 15000 samples
Feature: t_magnitude_sq Time: 7656 usec per 15000 samples
Feature: t_jerk Time: 6016 usec per 15000 samples
Feature: t_jerk+l1norm Time: 12186 usec per 15000 samples
Feature: t_jerk+magnitude_sq Time: 12500 usec per 15000 samples
"""

def get_processing_charges(results):
    print("Processing charges, uC per window")
    lines = results.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        fields = line.split()

        name = fields[1]
        tm = int(fields[3])
        num_samples = int(fields[6])
        num_samples *= 3 # because there are 3 axis used in the C code eval of all of the features & transforms

        num_windows = (num_samples / WINDOW_SIZE_SAMPLES) * utils.WINDOW_OVERLAP_TIMES

        #print("time=", tm, "num_windows=", num_windows)

        charge_uc = get_processing_charge_uc_per_time(tm)
        charge_uc_per_window = charge_uc / num_windows

        if DO_PRINT_AVG_CURRENT:
            seconds = SAMPLING_RATE / WINDOW_SIZE_SAMPLES
            avg_current = charge_uc_per_window / seconds
            print('costs_cpu["{}"] = {:f}\t{:f}\t{:f}'.format(name, charge_uc_per_window, tm / 1000.0, avg_current))
        else:
            print('costs_cpu["{}"] = {:f}'.format(name, charge_uc_per_window))

    print("")
        
##########################################

# This uses CBOR for data encoding
def stat_feature_cbor(data, key):
#    if key == "raw":
#        print("cbor:", key, len(data[key]) / NUM_PARTICIPANTS)
#    else:
#        print("cbor:", key, len(data[key]) / NUM_PARTICIPANTS * 64)
    num_bytes = 0
    lst = []
    last_bytes = 0
    for v in data[key]:
        lst.append(v)
        # try to compress; if fits in packet, use it
        cbor_blob = cbor.dumps(lst)
        if len(cbor_blob) > MAX_LEN:
            num_bytes += last_bytes
            lst = []
        else:
            last_bytes = len(cbor_blob)
    # add the final bit, if needed
    if len(lst):
        num_bytes += len(cbor.dumps(lst))
    return num_bytes

# this is just regular IEEE 4-byte floating point packaging
def plain_dumps(data, data_size):
    return " " * (len(data) * data_size)

def stat_feature_plain(data, key, data_size = 4):
    num_bytes = 0
    lst = []
    last_bytes = 0
    for v in data[key]:
        lst.append(v)
        # try to compress; if fits in packet, use it
        plain_blob = plain_dumps(lst, data_size)
        if len(plain_blob) > MAX_LEN:
            num_bytes += last_bytes
            lst = []
        else:
            last_bytes = len(plain_blob)
    # add the final bit, if needed
    if len(lst):
        num_bytes += len(plain_dumps(lst, data_size))
    return num_bytes

def stat_feature_plain_2b(data, key):
    return stat_feature_plain(data, key, 2)

def stat_feature_plain_4b(data, key):
    return stat_feature_plain(data, key, 4)

def stat_feature(data, key, fn):
    if fn:
        return fn(data, key)
    if len(data[key]) and type(data[key][0]) is float:
        #print("using float")
        # use 16-bit floats
        return stat_feature_plain_2b(data, key)
    # use cbor
    return stat_feature_cbor(data, key)

########################################

# Start feature: mean
def get_tx_charges(filename, rawdatadir):
    data = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or "Starting" in line or "Done" in line:
                continue
            if "Start feature" in line:
                name = line[15:]
                data[name] = []
                continue
            if "axis" in line:
                continue
            if "." in line:
                v = float(line)
            else:
                v = int(line)
            data[name].append(v)

    inputs = ["00001.c", "00002.c", "00003.c", "00004.c", "00005.c", "00007.c"]
    X = []
    Y = []
    Z = []
    for filename in inputs:
        with open(os.path.join(rawdatadir, filename), "r") as f:
            for line in f:
                line = line.strip()
                x, y, z = line.split(" ")
                x = int(x.strip("{").strip(" ").strip(","))
                y = int(y.strip(" ").strip(","))
                z = int(z.strip(" ").strip(",").strip("}"))
                X.append(x)
                Y.append(y)
                Z.append(z)
    data["raw"] = X + Y + Z

    num_samples_per_axis = len(X)
    num_windows = num_samples_per_axis / WINDOW_SIZE_SAMPLES * utils.WINDOW_OVERLAP_TIMES
    num_windows *= 3 # because there are 3 axis 

    print("Tx charges, uC per window")
    for name in data:
        #print(name, data[name])
        size = stat_feature(data, name, None)
        #print(name, "bytes per h", size)
        charge_uc = get_tx_charge_uc_per_bytes(size)

        charge_uc_per_window = charge_uc / num_windows
        if name == "raw":
            # because for raw data, there are no overalpping windows
            charge_uc_per_window /= utils.WINDOW_OVERLAP_TIMES

        if DO_PRINT_AVG_CURRENT:
            seconds = SAMPLING_RATE / WINDOW_SIZE_SAMPLES
            avg_current = charge_uc_per_window / seconds
            print('costs_tx["{}"] = {:f}\t{:f}'.format(name, charge_uc_per_window, avg_current))
        else:
            print('costs_tx["{}"] = {:f}'.format(name, charge_uc_per_window))

########################################

def main():
    get_processing_charges(results)
    get_tx_charges("out.txt", "../c-implementation/sample-data")

########################################

    
if __name__ == '__main__':
    main()
    print("all done!")
