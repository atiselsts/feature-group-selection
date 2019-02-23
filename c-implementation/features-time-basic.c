// -----------------------------------------------------------

void feature_nop(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            __asm__("nop");
        }
    }
}

// -----------------------------------------------------------

void feature_nop_nop(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            __asm__("nop");
            __asm__("nop");
        }
    }
}

// -----------------------------------------------------------

void feature_mean(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int32_t sum = 0;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            sum += data[i + j].v[axis];
        }

        int32_t avg = sum / TIME_WINDOW_SIZE;
        OUTPUT_I(avg, result_i.v[axis]);
        LOG("\n");
    }
}

// this is also known as root mean square
void feature_energy(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        uint32_t sqsum = 0;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            sqsum += (int)data[i + j].v[axis] * data[i + j].v[axis];
        }

        int32_t squared_avg = sqsum / TIME_WINDOW_SIZE;
        OUTPUT_F(tsqrtf(squared_avg), result_f.v[axis]);
        LOG("\n");
    }
}

void feature_energy_mean(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int32_t sum = 0;
        uint32_t sqsum = 0;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            sum += data[i + j].v[axis];
            sqsum += (int)data[i + j].v[axis] * data[i + j].v[axis];
        }

        int32_t avg = sum / TIME_WINDOW_SIZE;
        int32_t squared_avg = sqsum / TIME_WINDOW_SIZE;
        OUTPUT_I(avg, result_i.v[axis]);
        OUTPUT_F(tsqrtf(squared_avg), result_f.v[axis]);
        LOG("\n");
    }
}

void feature_std(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int32_t sum = 0;
        uint32_t sqsum = 0;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            sum += data[i + j].v[axis];
            sqsum += (int)data[i + j].v[axis] * data[i + j].v[axis];
        }

        int32_t avg = sum / TIME_WINDOW_SIZE;
        int32_t squared_avg = sqsum / TIME_WINDOW_SIZE;
        OUTPUT_F(tsqrtf(squared_avg - avg * avg), result_f.v[axis]);
        LOG("\n");
    }
}


void feature_std_mean(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int32_t sum = 0;
        uint32_t sqsum = 0;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            sum += data[i + j].v[axis];
            sqsum += (int)data[i + j].v[axis] * data[i + j].v[axis];
        }

        int32_t avg = sum / TIME_WINDOW_SIZE;
        int32_t squared_avg = sqsum / TIME_WINDOW_SIZE;
        OUTPUT_I(avg, result_i.v[axis]);
        OUTPUT_F(tsqrtf(squared_avg - avg * avg), result_f.v[axis]);
        LOG("\n");
    }
}

void feature_std_energy(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int32_t sum = 0;
        uint32_t sqsum = 0;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            sum += data[i + j].v[axis];
            sqsum += (int)data[i + j].v[axis] * data[i + j].v[axis];
        }

        int32_t avg = sum / TIME_WINDOW_SIZE;
        int32_t squared_avg = sqsum / TIME_WINDOW_SIZE;
        OUTPUT_F(tsqrtf(squared_avg), result_f.v[axis]);
        OUTPUT_F(tsqrtf(squared_avg - avg * avg), result_f.v[axis]);
        LOG("\n");
    }
}

void feature_std_energy_mean(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int32_t sum = 0;
        uint32_t sqsum = 0;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            sum += data[i + j].v[axis];
            sqsum += (int)data[i + j].v[axis] * data[i + j].v[axis];
        }

        int32_t avg = sum / TIME_WINDOW_SIZE;
        int32_t squared_avg = sqsum / TIME_WINDOW_SIZE;
        OUTPUT_I(avg, result_i.v[axis]);
        OUTPUT_F(tsqrtf(squared_avg), result_f.v[axis]);
        OUTPUT_F(tsqrtf(squared_avg - avg * avg), result_f.v[axis]);
        LOG("\n");
    }
}

// -----------------------------------------------------------

void feature_correlation(int axis)
{
    int i, j;
    int axis1 = axis;
    int axis2 = (axis + 1) % NUM_AXIS;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int32_t sum1 = 0, sum2 = 0;
        uint32_t sqsum1 = 0, sqsum2 = 0;
        int32_t msum = 0;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            sum1 += data[i + j].v[axis1];
            sqsum1 += (int)data[i + j].v[axis1] * data[i + j].v[axis1];

            sum2 += data[i + j].v[axis2];
            sqsum2 += (int)data[i + j].v[axis2] * data[i + j].v[axis2];

            msum += (int)data[i + j].v[axis1] * data[i + j].v[axis2];
        }

        int32_t avg1 = sum1 / TIME_WINDOW_SIZE;
        int32_t squared_avg1 = sqsum1 / TIME_WINDOW_SIZE;
        float std1 = tsqrtf(squared_avg1 - avg1 * avg1);

        int32_t avg2 = sum2 / TIME_WINDOW_SIZE;
        int32_t squared_avg2 = sqsum2 / TIME_WINDOW_SIZE;
        float std2 = tsqrtf(squared_avg2 - avg2 * avg2);

        int32_t avgm = msum / TIME_WINDOW_SIZE;

        float e = avgm - avg1 * avg2;
        float corr = (std1 == 0 || std2 == 0) ? 0 : e / (std1 * std2);

        OUTPUT_F(corr, result_f.v[axis]);

        LOG("\n");
    }
}

// -----------------------------------------------------------

void feature_correlation_std(int axis)
{
    int i, j;
    int axis1 = axis;
    int axis2 = (axis + 1) % NUM_AXIS;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int32_t sum1 = 0, sum2 = 0;
        uint32_t sqsum1 = 0, sqsum2 = 0;
        int32_t msum = 0;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            sum1 += data[i + j].v[axis1];
            sqsum1 += (int)data[i + j].v[axis1] * data[i + j].v[axis1];

            sum2 += data[i + j].v[axis2];
            sqsum2 += (int)data[i + j].v[axis2] * data[i + j].v[axis2];

            msum += (int)data[i + j].v[axis1] * data[i + j].v[axis2];
        }

        int32_t avg1 = sum1 / TIME_WINDOW_SIZE;
        int32_t squared_avg1 = sqsum1 / TIME_WINDOW_SIZE;
        float std1 = tsqrtf(squared_avg1 - avg1 * avg1);

        int32_t avg2 = sum2 / TIME_WINDOW_SIZE;
        int32_t squared_avg2 = sqsum2 / TIME_WINDOW_SIZE;
        float std2 = tsqrtf(squared_avg2 - avg2 * avg2);

        int32_t avgm = msum / TIME_WINDOW_SIZE;

        float e = avgm - avg1 * avg2;
        float corr = (std1 == 0 || std2 == 0) ? 0 : e / (std1 * std2);

        OUTPUT_F(corr, result_f.v[axis]);
        OUTPUT_F(std1, result_f.v[axis1]);

        LOG("\n");
    }
}

// -----------------------------------------------------------

void feature_correlation_std_std(int axis)
{
    int i, j;
    int axis1 = axis;
    int axis2 = (axis + 1) % NUM_AXIS;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        int32_t sum1 = 0, sum2 = 0;
        uint32_t sqsum1 = 0, sqsum2 = 0;
        int32_t msum = 0;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            sum1 += data[i + j].v[axis1];
            sqsum1 += (int)data[i + j].v[axis1] * data[i + j].v[axis1];

            sum2 += data[i + j].v[axis2];
            sqsum2 += (int)data[i + j].v[axis2] * data[i + j].v[axis2];

            msum += (int)data[i + j].v[axis1] * data[i + j].v[axis2];
        }

        int32_t avg1 = sum1 / TIME_WINDOW_SIZE;
        int32_t squared_avg1 = sqsum1 / TIME_WINDOW_SIZE;
        float std1 = tsqrtf(squared_avg1 - avg1 * avg1);

        int32_t avg2 = sum2 / TIME_WINDOW_SIZE;
        int32_t squared_avg2 = sqsum2 / TIME_WINDOW_SIZE;
        float std2 = tsqrtf(squared_avg2 - avg2 * avg2);

        int32_t avgm = msum / TIME_WINDOW_SIZE;

        float e = avgm - avg1 * avg2;
        float corr = (std1 == 0 || std2 == 0) ? 0 : e / (std1 * std2);

        OUTPUT_F(corr, result_f.v[axis]);
        OUTPUT_F(std1, result_f.v[axis1]);
        OUTPUT_F(std2, result_f.v[axis2]);

        LOG("\n");
    }
}

// -----------------------------------------------------------

void feature_sma(int axis)
{
    int i, j;
    LOG("axis=%d\n", axis);
    for (i = 0; i <= NSAMPLES - TIME_WINDOW_SIZE; i += PERIODIC_COMPUTATION_WINDOW_SIZE) {
        uint32_t abssum = 0;
        for (j = 0; j < TIME_WINDOW_SIZE; ++j) {
            abssum += abs(data[i + j].v[axis]);
        }

        OUTPUT_I(abssum / TIME_WINDOW_SIZE, result_i.v[axis]);
        LOG("\n");
    }
}

// -----------------------------------------------------------
