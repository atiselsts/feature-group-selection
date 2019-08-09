#
# This is similar to the Pandas function with the same name, but implemented here:
# 1) not all pc have pandas installed
# 2) the pandas function is slow
#
# By default, the lines are split on all whitespace characters. Pass sep=',' to split by comma (for example).
#
def load_csv(filename, sep = None, skiprows=0, parse_numbers=True):

    result = []
    c = 0 # non-empty line counter
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            c += 1
            if c <= skiprows:
                # skip the header lines
                continue

            fields = line.split(sep)
            record = []
            for f in fields:
                if parse_numbers:
                    try:
                        x = float(f) # as float
                    except:
                        x = f # as string
                else:
                    x = f # as string
                record.append(x)

            result.append(record)

    print("loaded file {}, dimensions: {} x {}".format(filename, len(result), len(result[0])))
    return result

###########################################

#
# This loads the list of features and groups then in categories
#
def read_list_of_features(filename, filters=None):
    names = []
    with open(filename, "r") as f:
        line_number = 0
        for line in f:
            line = line.strip()
            if line == "":
                continue
            line_number += 1
            name = line
            # skip these: not present in general, as most datasets don't have gyros
            if "Gyro" in name or "angle" in name:
                continue
            # skip these: too difficult to transmit
            if "bandsEnergy" in name:
                continue

            group_name = "-".join(name.split('-')[:2])

            if filters:
                if not any(f in group_name for f in filters):
                    # does not match any filters; skip
                    continue

            if "-" in group_name:
                category = name.split('-')[0]
                function = name.split('-')[1]
            else:
                category = name
                function = ""

            names.append([line_number - 1, name, group_name, category, function])

    return names

###########################################

#
# This results as list of numbers corresponding to `used_indexes`
#
def select(names, groups, used_indexes, do_subselection = False):
    result = []
    #print("names=")
    #for n in names:
    #   print("  ", n)

    for i in used_indexes:
        group = groups[i]
        #print('group=', group)
        for number, name, group_name, category, function in names:
            if do_subselection:
                if name == group:
                    result.append(number)
            else:
                if group_name == group:
                    result.append(number)
    #print("select", used_indexes, ":", result)
    return result

###########################################

ALL_DATASETS = ["UCI HAR Dataset", "SPHERE", "PAMAP2"]

ALL_DATASETS_SHORT = ["har", "sphere", "pamap2"]

# keep this the same for all
WINDOW_SIZE_SAMPLES = 128

# "2" means 50% overlap between subsequent windows
WINDOW_OVERLAP_TIMES = 2
