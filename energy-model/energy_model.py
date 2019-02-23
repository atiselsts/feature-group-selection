#!/usr/bin/python3

############################################

NUM_AXIS = 3

# Apply median filter before working with the data?
DO_MEDIAN_FILTER = True

############################################

# idea:
# - account for pass-through complexity once and remove this pass-through cost from each subsequenct feature
# - account for each of transformation just once
# - account for each feature:
# -- if xyz features are used, multiply the cost of the feature by 3
#

############################################

# energy costs are given in uC, per one 128 sample windows
# these costs are per processing a single axis of data!
costs_cpu = {}
costs_cpu["nop"] = 0.012402
costs_cpu["nop_nop"] = 0.014724
costs_cpu["mean"] = 0.026026
costs_cpu["energy"] = 0.032430
costs_cpu["energy+mean"] = 0.035544
costs_cpu["std"] = 0.035329
costs_cpu["std+mean"] = 0.035444
costs_cpu["std+energy"] = 0.039442
costs_cpu["std+energy+mean"] = 0.039838
costs_cpu["correlation"] = 0.066583
costs_cpu["correlation+std"] = 0.066467
costs_cpu["correlation+std+std"] = 0.066583
costs_cpu["entropy"] = 0.257281
costs_cpu["min"] = 0.026026
costs_cpu["min+max"] = 0.030931
costs_cpu["median"] = 0.064468
costs_cpu["iqr"] = 0.069977
costs_cpu["median+iqr"] = 0.081483
costs_cpu["median+iqr+min+max"] = 0.091505
costs_cpu["t_median"] = 0.033430
costs_cpu["t_l1norm"] = 0.034237
costs_cpu["t_magnitude_sq"] = 0.029432
costs_cpu["t_jerk"] = 0.023127
costs_cpu["t_jerk+l1norm"] = 0.046846
costs_cpu["t_jerk+magnitude_sq"] = 0.048053

# equal features
costs_cpu["max"] = costs_cpu["min"]
costs_cpu["q25"] = costs_cpu["median"]
costs_cpu["q75"] = costs_cpu["median"]

# the cost of running just the loop itself
costs_cpu["empty_loop"] = costs_cpu["nop"] - (costs_cpu["nop_nop"] - costs_cpu["nop"])


# These costs are also assuming the same settings, efficient CBOR encoding (16bit floats),
# and in uC, for a single window, single axis of data
costs_tx = {}
costs_tx["mean"] = 0.894657
costs_tx["energy"] = 1.488759
costs_tx["std"] = 1.488759
costs_tx["correlation"] = 1.488759
costs_tx["entropy"] = 1.488759
costs_tx["min"] = 1.015717
costs_tx["max"] = 1.172275
costs_tx["median"] = 1.016627
costs_tx["q25"] = 1.016627
costs_tx["q75"] = 1.016718
costs_tx["iqr"] = 0.843412
costs_tx["raw"] = 31.460966

############################################

def account_transforms(fs):
    # median
    # jerk
    # magn

    # all data is first processed by the "median of three" transform
    r = costs_cpu["t_median"]

    # then, some data is processed by the jerk transform, on top of which l1norm or mag_sq can be applied

    if any(1 for f in fs if "Jerk" in f):
        if any(1 for f in fs if "MagSq" in f):
            r += costs_cpu["t_jerk+magnitude_sq"]
        elif any(1 for f in fs if "L1Norm" in f):
            r += costs_cpu["t_jerk+l1norm"]
        else:
            r += costs_cpu["t_jerk"]
    else:
        # no jerk; just look for l1norm and mag_sq transforms
        if any(1 for f in fs if "MagSq" in f):
            r += costs_cpu["t_magnitude_sq"]
        elif any(1 for f in fs if "L1Norm" in f):
            r += costs_cpu["t_l1norm"]
    return r

def remove_correlation(fs):
    # Feature: correlation Time: 17290 usec per 15000 samples
    # Feature: correlation+std Time: 17266 usec per 15000 samples
    # Feature: correlation+std+std Time: 17423 usec per 15000 samples
    if not any(1 for f in fs if "correlation" in f):
        return 0, fs
    fsnew = []
    for f in fs:
        if not ("std" in f or "mean" in f or "energy" in f or "correlation" in f):
            fsnew.append(f)
    r = costs_cpu["correlation"] # good enough approximation
    return r, fsnew

def remove_std(fs):
    # Feature: std Time: 9193 usec per 15000 samples
    # Feature: std+mean Time: 9216 usec per 15000 samples
    # Feature: std+energy Time: 9846 usec per 15000 samples
    # Feature: std+energy+mean Time: 8436 usec per 15000 samples
    if not any(1 for f in fs if "std" in f):
        return 0, fs
    fsnew = []
    for f in fs:
        if not ("std" in f or "mean" in f or "energy" in f):
            fsnew.append(f)
    r = costs_cpu["std"] # good enough approximation
    return r, fsnew

def remove_energy(fs):
    # Feature: mean Time: 6640 usec per 15000 samples
    # Feature: energy Time: 6666 usec per 15000 samples
    # Feature: energy+mean Time: 6770 usec per 15000 samples
    if not any(1 for f in fs if "energy" in f):
        return 0, fs
    fsnew = []
    for f in fs:
        if not ("mean" in f or "energy" in f):
            fsnew.append(f)
    r = costs_cpu["energy"] # good enough approximation
#    print("fsnew", fsnew)
    return r, fsnew

def remove_median(fs):
    # these are the results:
    # Feature: min Time: 6643 usec per 15000 samples
    # Feature: min+max Time: 7916 usec per 15000 samples
    # Feature: median Time: 16770 usec per 15000 samples
    # Feature: iqr Time: 18226 usec per 15000 samples
    # Feature: median+iqr Time: 21170 usec per 15000 samples
    # Feature: median+iqr+min+max Time: 23776 usec per 15000 samples
    has_iqr = False
    has_iqr = False
    has_max = False
    has_median = False
    has_min = False
    fsnew = []
    for f in sorted(fs):
        if "iqr" in f:
            has_iqr = True
        elif "max" in f:
            has_max = True
        elif "min" in f:
            has_min = True
        elif "median" in f:
            has_min = True
        elif "q25" in f:
            has_median = True # ~same complexity as median
        elif "q75" in f:
            has_median = True # ~same complexity as median
        else:
            fsnew.append(f)

    r = 0
    if has_median and has_iqr:
        if has_min or has_max:
            r = costs_cpu["median+iqr+min+max"]
        else:
            r = costs_cpu["median+iqr"]
    else:
        if has_min and has_max:
            r = costs_cpu["min+max"]
        elif has_min or has_max:
            r = costs_cpu["min"]
        if has_median:
            r += costs_cpu["median"]
        if has_iqr:
            r += costs_cpu["iqr"]
    return r, fsnew

def separate_by_prefix(fs):
    normal = []
    l1norm = []
    magsq = []
    jerk = []
    jerkl1norm = []
    jerkmagsq = []
    for f in fs:
        if "JerkMagSq" in f:
            jerkmagsq.append(f)
        elif "MagSq" in f:
            magsq.append(f)
        elif "JerkL1Norm" in f:
            jerkl1norm.append(f)
        elif "L1Norm" in f:
            l1norm.append(f)
        elif "Jerk" in f:
            jerk.append(f)
        else:
            normal.append(f)
    return normal, l1norm, magsq, jerk, jerkl1norm, jerkmagsq

def is_multiaxial(f):
    if "L1Norm" in f or "Mag" in f or "MagSq" in f:
        # single axis
        return False
    return True

def extractType(f):
    ftype = f.split("-")[1]
    return ftype[:-2]

def calc(fs):
    # Takes into account the pass-through processing
    adjust()

    # Transmission
    total_tx_cost = 0
    for f in fs:
        ftype = extractType(f)
        if ftype in costs_tx:
            c = costs_tx[ftype]
            if is_multiaxial(f):
                c *= NUM_AXIS
        else:
            print("unknown type:", ftype)
            c = 0

        total_tx_cost += c

    # CPU
    total_cpu_cost = costs_cpu["empty_loop_saved"]

    # account for all of the transforms needed for the data
    total_cpu_cost += account_transforms(fs)

    #print("cost of empty and transforms", costs_cpu["empty_loop_saved"], account_transforms(fs))

    for fsi in separate_by_prefix(fs):
        if len(fsi) == 0:
            continue

        c, fsi = remove_correlation(fsi)
        total_cpu_cost += c

        c, fsi = remove_std(fsi)
        total_cpu_cost += c

        c, fsi = remove_energy(fsi)
        total_cpu_cost += c

        c, fsi = remove_median(fsi)
        total_cpu_cost += c

        # deal with the remaining features not in any of the previous classes
        for f in fsi:
            ftype = extractType(f)
            #print("account for ", f)
            c = costs_cpu[ftype]
            if is_multiaxial(f):
                c *= NUM_AXIS
            total_cpu_cost += c
    
    return total_cpu_cost, total_tx_cost

def calc_raw():
    adjust()
    # just the raw data
    return costs_cpu["empty_loop_saved"], costs_tx["raw"]


#####################################################

def adjust():
    # make this method idempotent
    if costs_cpu["empty_loop"]:
        # remove the cost of the loop itself from all of the features
        # (it could be though of as the cost of sampling)
        for key in costs_cpu:
            if key != "empty_loop":
                costs_cpu[key] -= costs_cpu["empty_loop"]
        costs_cpu["empty_loop_saved"] = costs_cpu["empty_loop"]
        costs_cpu["empty_loop"] = 0

def test():
    print("\ntest")

    lst = []
    print(calc(lst), lst)

    lst = [
        "tTotalAcc-mean()"
    ]
    print(calc(lst), lst)

    lst = [
        "tTotalAccJerk-mean()"
    ]
    print(calc(lst), lst)

    lst = [
        "tTotalAccMagSq-mean()"
    ]
    print(calc(lst), lst)

    lst = [
        "tTotalAccMagSq-mean()",
        "tTotalAccJerkMagSq-mean()"
    ]
    print(calc(lst), lst)

    lst = [
        "tTotalAccMag-mean()",
        "tTotalAccJerkMag-mean()"
    ]
    print(calc(lst), lst)

    lst = [
        "tTotalAccMagSq-max()",
    ]
    print(calc(lst), lst)

    lst = [
        "tTotalAccMagSq-max()",
        "tTotalAccMagSq-max()",
    ]
    print(calc(lst), lst)

    lst = [
        "tTotalAcc-max()",
        "tTotalAccMagSq-max()",
    ]
    print(calc(lst), lst)

    lst = [
        "tTotalAcc-mean()",
        "tTotalAcc-std()",
        "tTotalAcc-max()",
        "tTotalAcc-min()",
        "tTotalAcc-energy()",
        "tTotalAcc-iqr()"
    ]
    print(calc(lst), lst)

#####################################################

#test()
def do_simple():
    print("\ndo_simple")
    print(calc_raw(), 'for raw data')

def do_all():
    print("\ndo_all")
    lst = [
        "tTotalAcc-mean()-X", "tTotalAcc-mean()-Y", "tTotalAcc-mean()-Z", "tTotalAcc-max()-X", "tTotalAcc-max()-Y", "tTotalAcc-max()-Z", "tTotalAcc-min()-X", "tTotalAcc-min()-Y", "tTotalAcc-min()-Z", "tTotalAcc-median()-X", "tTotalAcc-median()-Y", "tTotalAcc-median()-Z", "tTotalAcc-iqr()-X", "tTotalAcc-iqr()-Y", "tTotalAcc-iqr()-Z", "tTotalAcc-energy()-X", "tTotalAcc-energy()-Y", "tTotalAcc-energy()-Z", "tTotalAcc-std()-X", "tTotalAcc-std()-Y", "tTotalAcc-std()-Z", "tTotalAcc-correlation()-XY tTotalAcc-correlation()-XZ tTotalAcc-correlation()-YZ tTotalAcc-entropy()-X", "tTotalAcc-entropy()-Y", "tTotalAcc-entropy()-Z", "tTotalAccL1Norm-mean()", "tTotalAccL1Norm-min()", "tTotalAccL1Norm-max()", "tTotalAccL1Norm-median()", "tTotalAccL1Norm-iqr()", "tTotalAccL1Norm-energy()", "tTotalAccL1Norm-std()", "tTotalAccL1Norm-entropy()", "tTotalAccMagSq-mean()", "tTotalAccMagSq-min()", "tTotalAccMagSq-max()", "tTotalAccMagSq-median()", "tTotalAccMagSq-iqr()", "tTotalAccMagSq-energy()", "tTotalAccMagSq-std()", "tTotalAccMagSq-entropy()", "tTotalAccJerk-mean()-X", "tTotalAccJerk-mean()-Y", "tTotalAccJerk-mean()-Z", "tTotalAccJerk-max()-X", "tTotalAccJerk-max()-Y", "tTotalAccJerk-max()-Z", "tTotalAccJerk-min()-X", "tTotalAccJerk-min()-Y", "tTotalAccJerk-min()-Z", "tTotalAccJerk-median()-X", "tTotalAccJerk-median()-Y", "tTotalAccJerk-median()-Z", "tTotalAccJerk-iqr()-X", "tTotalAccJerk-iqr()-Y", "tTotalAccJerk-iqr()-Z", "tTotalAccJerk-energy()-X", "tTotalAccJerk-energy()-Y", "tTotalAccJerk-energy()-Z", "tTotalAccJerk-std()-X", "tTotalAccJerk-std()-Y", "tTotalAccJerk-std()-Z", "tTotalAccJerk-correlation()-XY tTotalAccJerk-correlation()-XZ tTotalAccJerk-correlation()-YZ tTotalAccJerk-entropy()-X", "tTotalAccJerk-entropy()-Y", "tTotalAccJerk-entropy()-Z", "tTotalAccJerkL1Norm-mean()", "tTotalAccJerkL1Norm-min()", "tTotalAccJerkL1Norm-max()", "tTotalAccJerkL1Norm-median()", "tTotalAccJerkL1Norm-iqr()", "tTotalAccJerkL1Norm-energy()", "tTotalAccJerkL1Norm-std()", "tTotalAccJerkL1Norm-entropy()", "tTotalAccJerkMagSq-mean()", "tTotalAccJerkMagSq-min()", "tTotalAccJerkMagSq-max()", "tTotalAccJerkMagSq-median()", "tTotalAccJerkMagSq-iqr()", "tTotalAccJerkMagSq-energy()", "tTotalAccJerkMagSq-std()", "tTotalAccJerkMagSq-entropy()"
    ]
    print(calc(lst), 'for all')

def do_greedy():
    print("\ndo_greedy")
    lst = ['tTotalAcc-q25()']
    print(sum(calc(lst)))
    lst = ['tTotalAcc-q25()', 'tTotalAccJerkMagSq-iqr()']
    print(sum(calc(lst)))
    lst = ['tTotalAcc-median()', 'tTotalAcc-min()', 'tTotalAccJerkMagSq-iqr()']
    print(sum(calc(lst)))
    lst = ['tTotalAcc-median()', 'tTotalAcc-max()', 'tTotalAcc-min()', 'tTotalAccJerkMagSq-iqr()']
    print(sum(calc(lst)))
    lst = ['tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()', 'tTotalAcc-min()', 'tTotalAcc-median()', 'tTotalAccJerk-energy()']
    print(sum(calc(lst)))

def do_greedy2():
    print("\ndo_greedy2")
    lst = ['tTotalAcc-max()']
    print(sum(calc(lst)))
    lst = ['tTotalAcc-max()', 'tTotalAcc-correlation()']
    print(sum(calc(lst)))
    lst = ['tTotalAccMag-median()', 'tTotalAcc-max()', 'tTotalAcc-correlation()']
    print(sum(calc(lst)))
    lst = ['tTotalAcc-max()', 'tTotalAcc-energy()', 'tTotalAccMag-median()', 'tTotalAcc-correlation()']
    print(sum(calc(lst)))
    lst = ['tTotalAcc-max()', 'tTotalAcc-energy()', 'tTotalAccJerkMag-std()', 'tTotalAccMag-median()', 'tTotalAcc-correlation()']
    print(sum(calc(lst)))
    lst = ['tTotalAcc-max()', 'tTotalAcc-energy()', 'tTotalAccJerkMag-std()', 'tTotalAccJerk-max()', 'tTotalAccMag-median()', 'tTotalAcc-correlation()']
    print(sum(calc(lst)))
    lst =  ['tTotalAcc-max()', 'tTotalAcc-energy()', 'tTotalAccJerkMag-std()', 'tTotalAccJerk-max()', 'tTotalAccJerkMagSq-entropy()', 'tTotalAccMag-median()', 'tTotalAcc-correlation()']
    print(sum(calc(lst)))

def do_specific():
    print("\ndo_speficic")

    features = ['tTotalAcc-std()',
                'tTotalAcc-entropy()',
                'tTotalAcc-energy()',
                'tTotalAccJerk-entropy()',
                'tTotalAccJerk-std()',
                'tTotalAcc-mean()',
                'tTotalAccJerk-energy()',
                'tTotalAcc-correlation()',
                'tTotalAcc-max()']     

    for i in range(1, len(features) + 1):
        lst = features[:i]
        print(sum(calc(lst)))
        print(lst)

#####################################################

def main():
    
    test()
    do_simple()
    do_all()
    do_greedy()
    do_greedy2()
    do_specific()

###########################################
    
if __name__ == '__main__':
    main()
    print("all done!")
