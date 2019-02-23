#include <stdint.h>

#ifndef MAIN_H
#define MAIN_H

#define SAMPLING_HZ 20

#ifndef TIME_WINDOW_SIZE
#define TIME_WINDOW_SIZE 128
#endif

#if TIME_WINDOW_SIZE == 32
#pragma message "TIME_WINDOW_SIZE == 32"
#elif TIME_WINDOW_SIZE == 64
#pragma message "TIME_WINDOW_SIZE == 64"
#elif TIME_WINDOW_SIZE == 128
#pragma message "TIME_WINDOW_SIZE == 128"
#endif

// Make the windows to have 50% overlap
#define PERIODIC_COMPUTATION_WINDOW_SIZE (TIME_WINDOW_SIZE / 2)

#define VERY_FAST 0
#define FAST 1
#define MODERATE 2
#define SLOW 3

// If this is set to true, repetitions are not done;
// instead, the output is logged for each.
#ifndef DO_LOG_OUTPUT
#define DO_LOG_OUTPUT 0
#endif

#if DO_LOG_OUTPUT

static const int NUM_REPETITIONS[4] = {
    [VERY_FAST] = 1,
    [FAST] = 1,
    [MODERATE] = 1,
    [SLOW] = 1
};

#else // DO_LOG_OUTPUT

#if CONTIKI_TARGET_SRF06_CC26XX || CONTIKI_TARGET_ZOUL || CONTIKI_TARGET_NRF52DK
static const int NUM_REPETITIONS[4] = {
    [VERY_FAST] = 100,
    [FAST] = 10,
    [MODERATE] = 1,
    [SLOW] = 1
};
#elif CONTIKI_TARGET_Z1
static const int NUM_REPETITIONS[4] = {
    [VERY_FAST] = 10,
    [FAST] = 10,
    [MODERATE] = 1,
    [SLOW] = 1
};
#else
static const int NUM_REPETITIONS[4] = {
    [VERY_FAST] = 1000,
    [FAST] = 1000,
    [MODERATE] = 100,
    [SLOW] = 10
};
#endif
#endif // DO_LOG_OUTPUT

// -----------------------------------------------------------

#if DO_LOG_OUTPUT
#define LOG(...) printk(__VA_ARGS__)
#else
#define LOG(...)
#endif

#if DO_LOG_OUTPUT
#define OUTPUT(x, variable, format) printk(format, x)
#else
#define OUTPUT(x, variable, format) variable = x
#endif

#define OUTPUT_I(x, variable)  OUTPUT(x, variable, "%d ")
#define OUTPUT_IL(x, variable) OUTPUT(x, variable, "%lld ")
#define OUTPUT_F(x, variable)  OUTPUT(x, variable, "%f ")

#define NUM_AXIS 3

// -----------------------------------------------------------

typedef struct {
    int8_t v[NUM_AXIS];
} accel_t;

typedef struct {
    volatile int32_t v[NUM_AXIS];
} result_i_t;

typedef struct {
    volatile float v[NUM_AXIS];
} result_f_t;

// -----------------------------------------------------------

// the total number of samples
#define NSAMPLES ((unsigned int)(sizeof(data) / sizeof(*data)))

// -----------------------------------------------------------

typedef void (*feature_function)(int);

// the result of the accel calculations is stored here
volatile result_i_t result_i;
volatile result_f_t result_f;

// -----------------------------------------------------------


#endif
