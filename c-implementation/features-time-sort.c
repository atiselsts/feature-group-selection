// -----------------------------------------------------------

void feature_min(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i < NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int minval = INT_MAX;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            minval = min(minval, data[i + j].v[axis]);
        }
        OUTPUT_I(minval, result_i.v[axis]);
        LOG("\n");
    }
}

// -----------------------------------------------------------

void feature_max(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i < NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int maxval = INT_MIN;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            maxval = max(maxval, data[i + j].v[axis]);
        }
        OUTPUT_I(maxval, result_i.v[axis]);
        LOG("\n");
    }
}

// -----------------------------------------------------------

void feature_min_max(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i < NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int minval = INT_MAX;
        int maxval = INT_MIN;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            minval = min(minval, data[i + j].v[axis]);
            maxval = max(maxval, data[i + j].v[axis]);
        }
        OUTPUT_I(minval, result_i.v[axis]);
        OUTPUT_I(maxval, result_i.v[axis]);
        LOG("\n");
    }
}

// -----------------------------------------------------------

void feature_select_nth(int axis, int nth)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i < NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        // put all data in bins and walk through the bins while the nth element is found
        int8_t stats[256] = {0};
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            int v = data[i + j].v[axis] + 128;
            stats[v]++;
        }
        for (j = 0; j < 256; ++j) {
            if(stats[j] >= nth) break;
            nth -= stats[j];
        }        
        OUTPUT_I(j - 128, result_i.v[axis]);
        LOG("\n");
    }
}

void feature_q25(int axis)
{
    feature_select_nth(axis, TIME_WINDOW_SIZE / 4);
}

void feature_median(int axis)
{
    feature_select_nth(axis, TIME_WINDOW_SIZE / 2);
}

void feature_q75(int axis)
{
    feature_select_nth(axis, TIME_WINDOW_SIZE *  3 / 4);
}

// -----------------------------------------------------------

void feature_iqr(int axis)
{
    int i, j;
    int q25 = 0, q75 = 0;

    LOG("axis=%d\n", axis);
    for (i = 0; i < NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        // put all data in bins and walk through the bins while the nth element is found
        int8_t stats[256] = {0};
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            int v = data[i + j].v[axis] + 128;
            stats[v]++;
        }
        int c = 0;
        bool q25_set = false;
        bool q75_set = false;
        for (j = 0; j < 256; ++j) {
            if (stats[j]) {
                c += stats[j];
                if (c >= TIME_WINDOW_SIZE / 4 &&  !q25_set) {
                    q25_set = true;
                    q25 = j - 128;
                }
                if (c >= TIME_WINDOW_SIZE * 3 / 4 && !q75_set) {
                    q75_set = true;
                    q75 = j - 128;
                    break;
                }
            }
        }
        OUTPUT_I(q75 - q25, result_i.v[axis]);
        LOG("\n");
    }
}

// -----------------------------------------------------------

void feature_median_iqr(int axis)
{
    int i, j;
    int median = 0, q25 = 0, q75 = 0;

    LOG("axis=%d\n", axis);
    for (i = 0; i < NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        // put all data in bins and walk through the bins while the nth element is found
        int8_t stats[256] = {0};
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            int v = data[i + j].v[axis] + 128;
            stats[v]++;
        }
        int c = 0;
        bool median_set = false;
        bool q25_set = false;
        bool q75_set = false;
        for (j = 0; j < 256; ++j) {
            if (stats[j]) {
                c += stats[j];
                if(c >= TIME_WINDOW_SIZE / 2 && !median_set) {
                    median_set = true;
                    median = j - 128;
                }
                if (c >= TIME_WINDOW_SIZE / 4 &&  !q25_set) {
                    q25_set = true;
                    q25 = j - 128;
                }
                if (c >= TIME_WINDOW_SIZE * 3 / 4 && !q75_set) {
                    q75_set = true;
                    q75 = j - 128;
                    break;
                }
            }
        }
        OUTPUT_I(median, result_i.v[axis]);
        OUTPUT_I(q75 - q25, result_i.v[axis]);
        LOG("\n");
    }
}

// -----------------------------------------------------------

void feature_median_iqr_min_max(int axis)
{
    int i, j;
    int median = 0, q25 = 0, q75 = 0;

    LOG("axis=%d\n", axis);
    for (i = 0; i < NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        // put all data in bins and walk through the bins while the nth element is found
        int8_t stats[256] = {0};
        int minval = INT_MAX;
        int maxval = INT_MIN;       
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            minval = min(minval, data[i + j].v[axis]);
            maxval = max(maxval, data[i + j].v[axis]);

            int v = data[i + j].v[axis] + 128;
            stats[v]++;
        }
        int c = 0;
        bool median_set = false;
        bool q25_set = false;
        bool q75_set = false;
        for (j = 0; j < 256; ++j) {
            if (stats[j]) {
                c += stats[j];
                if(c >= TIME_WINDOW_SIZE / 2 && !median_set) {
                    median_set = true;
                    median = j - 128;
                }
                if (c >= TIME_WINDOW_SIZE / 4 &&  !q25_set) {
                    q25_set = true;
                    q25 = j - 128;
                }
                if (c >= TIME_WINDOW_SIZE * 3 / 4 && !q75_set) {
                    q75_set = true;
                    q75 = j - 128;
                    break;
                }
            }
        }       
        OUTPUT_I(median, result_i.v[axis]);
        OUTPUT_I(q75 - q25, result_i.v[axis]);
        OUTPUT_I(minval, result_i.v[axis]);
        OUTPUT_I(maxval, result_i.v[axis]);
        LOG("\n");
    }
}

// -----------------------------------------------------------

int cmp(const void *v1, const void *v2)
{
    const int8_t *x1 = v1;
    const int8_t *x2 = v2;
    return (int)*x1 - (int)*x2;
}

void feature_sort_median(int axis)
{
    int i, j;
    int8_t buffer[TIME_WINDOW_SIZE] = {0};
    LOG("axis=%d\n", axis);
    for (i = 0; i < NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            buffer[j] = data[i + j].v[axis];
        }
        qsort(buffer, TIME_WINDOW_SIZE, 1, cmp);

        int median = buffer[TIME_WINDOW_SIZE / 2];
        OUTPUT_I(median, result_i.v[axis]);
        LOG("\n");
    }
}

void feature_sort_iqr(int axis)
{
    int i, j;
    int8_t buffer[TIME_WINDOW_SIZE] = {0};
    LOG("axis=%d\n", axis);
    for (i = 0; i < NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            buffer[j] = data[i + j].v[axis];
        }
        qsort(buffer, TIME_WINDOW_SIZE, 1, cmp);

        int iqr = buffer[3 * TIME_WINDOW_SIZE / 4] - buffer[TIME_WINDOW_SIZE / 4];
        OUTPUT_I(iqr, result_i.v[axis]);
        LOG("\n");
    }
}

// -----------------------------------------------------------
