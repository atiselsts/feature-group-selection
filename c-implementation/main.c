#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

#include "adaptation.h"
#include "sqrt.h"
#include "main.h"

// -----------------------------------------------------------

// the input data
#if CONTIKI_TARGET_Z1
// take only 7500 samples: 10000 or more are not supported by the compiler
const accel_t data[7500] =
#else
// take as many samples as provided
const accel_t data[] =
#endif
{
# include "sample-data/00001-1.c"
};

// -----------------------------------------------------------

#include "features-time-basic.c"
#include "features-time-sort.c"
#include "features-time-advanced.c"
#include "transforms.c"

// -----------------------------------------------------------

typedef void (*feature_function)(int);

typedef struct {
    const char *name;
    feature_function f;
    int class;
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
        for (i = 0; i < NUM_REPETITIONS[VERY_FAST]; ++i) {
            t->f(axis);
        }
    }

    // print time needed
    delta = k_uptime_delta(&start);

#if 1
    printk("Feature: %s Time: %lu usec per %u samples\n",
            t->name,
            (long unsigned)(delta * 1000.0 / (NUM_AXIS * NUM_REPETITIONS[t->class])),
            NSAMPLES);
#else
    printk("Feature: %s Time: %lld ms (%d times)\n",
            t->name, delta, NUM_REPETITIONS[t->class]);
#endif
}

// -----------------------------------------------------------

const test_t tests[] =
{
    // Empty loops
    { "nop", feature_nop },
    { "nop_nop", feature_nop_nop },

    // Mean + energy + std combination
    { "mean", feature_mean },
    { "energy", feature_energy },
    { "energy+mean", feature_energy_mean },
    { "std", feature_std },
    { "std+mean", feature_std_mean },
    { "std+energy", feature_std_energy },
    { "std+energy+mean", feature_std_energy_mean },

    // Correlation + std combination
    { "correlation", feature_correlation },
    { "correlation+std", feature_correlation_std },
    { "correlation+std+std", feature_correlation_std_std },

    // Entropy
    { "entropy", feature_entropy },

    // Sorting-related functions
    { "min", feature_min },
    { "min+max", feature_min_max },
    { "median", feature_median },
    { "iqr", feature_iqr },
    { "median+iqr", feature_median_iqr },
    { "median+iqr+min+max", feature_median_iqr_min_max },

    // transforms
    { "t_median", transform_median }, /* this is kind of implicit before any other features are calculated */
    { "t_l1norm", transform_l1norm },
    { "t_magnitude_sq", transform_magnitude_sq },

    { "t_jerk", transform_jerk },
    { "t_jerk+l1norm", transform_jerk_l1norm },
    { "t_jerk+magnitude_sq", transform_jerk_magnitude_sq },
};

// -----------------------------------------------------------

void do_tests(void)
{
    int i;

    printk("Starting tests, ARCH=%s F_CPU=%d MHz\n",
           CONFIG_ARCH, (int)(CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC / 1000000));
    for (i = 0; i < sizeof(tests) / sizeof(*tests); ++i) {
        test(&tests[i]);
#if CONTIKI_TARGET_SRF06_CC26XX
        // don't let the watchdog expire
        hw_watchdog_periodic();
#endif
    }
    printk("Done!\n");
}

// -----------------------------------------------------------

#if CONTIKI

#include "dev/watchdog.h"

PROCESS(test_process, "Test process");
AUTOSTART_PROCESSES(&test_process);

PROCESS_THREAD(test_process, ev, data)
{
  PROCESS_BEGIN();

  printk("start\n");

#if CONTIKI_TARGET_NRF52DK
  // Enable the FPU bits in the Coprocessor Access Control Register
  SCB->CPACR |= (3UL << 20) | (3UL << 22);
  // Data Synchronization Barrier
  __DSB();
  // Instruction Synchronization Barrier
  __ISB();
#else // CONTIKI_TARGET_NRF52DK

  // Stop the watchdog (breaks the system on NRF52DK).
  // Note: this function is used as not all platforms define watchdog_stop()!
  watchdog_init();

#endif
 
  do_tests();

  PROCESS_END();
}

#else

int main(void)
{
    do_tests();
}

#endif

// -----------------------------------------------------------
