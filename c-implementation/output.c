#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

#ifdef CONFIG_ARCH
#include <zephyr.h>
#include <misc/printk.h>
#include "qsort.h"
#else
#include "adaptation.h"
#endif

#include "sqrt.h"

#define DO_LOG_OUTPUT 1
#include "main.h"

// -----------------------------------------------------------

// the input data
const accel_t data[] =
{
#include "sample-data/00001.c"
    ,
#include "sample-data/00002.c"
    ,
#include "sample-data/00003.c"
    ,
#include "sample-data/00004.c"
    ,
#include "sample-data/00005.c"
    ,
#include "sample-data/00007.c"
};

// -----------------------------------------------------------

#include "features-time-basic.c"
#include "features-time-sort.c"
#include "features-time-advanced.c"

// -----------------------------------------------------------

typedef struct {
    const char *name;
    feature_function f;
} test_t;

// -----------------------------------------------------------

void test(const test_t *t)
{
    int i, axis;
    s64_t start, delta;

    start = k_uptime_get();
    LOG("Start feature: %s\n", t->name);
    // iterate for each axis
    for (axis = 0; axis < NUM_AXIS; ++axis) {
        t->f(axis);
    }
}

// -----------------------------------------------------------

const test_t tests[] =
{
    // Mean + energy + std combination
    { "mean", feature_mean },
    { "energy", feature_energy },
    { "std", feature_std },

    // Correlation + std combination
    { "correlation", feature_correlation },

    // Entropy
    { "entropy", feature_entropy },

    // Sorting-related functions
    { "min", feature_min },
    { "max", feature_max },
    { "median", feature_median },
    { "q25", feature_q25 },
    { "q75", feature_q75 },
    { "iqr", feature_iqr },

    // SMA (sum of absolute values)
    //{ "sma", feature_sma },
};

// -----------------------------------------------------------

void do_tests(void)
{
    int i;

    printk("Starting tests, ARCH=%s F_CPU=%d MHz\n",
           CONFIG_ARCH, (int)(CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC / 1000000));

    for (i = 0; i < sizeof(tests) / sizeof(*tests); ++i) {
        test(&tests[i]);
    }

    printk("Done!\n");
}

// -----------------------------------------------------------

int main(void)
{
    do_tests();
}


// -----------------------------------------------------------
