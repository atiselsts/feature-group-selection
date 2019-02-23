//
// This is an adaptation layer for functinality normally provided by the Zephyr OS
//

#include <sys/time.h>
#include <stdio.h>
#include <stdio.h>
#include <math.h>

#define printk printf

typedef long long int s64_t;

#if CONTIKI_TARGET_SRF06_CC26XX || CONTIKI_TARGET_Z1 || CONTIKI_TARGET_ZOUL || CONTIKI_TARGET_NRF52DK
#define CONTIKI 1
#else
#define CONTIKI 0
#endif

#if CONTIKI
#include "contiki.h"

#define CONFIG_ARCH "sphere"
#define CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC (48u * 1000 * 1000)

static inline s64_t k_uptime_get(void)
{
    unsigned long t = clock_time();
    // convert to milliseconds
    return t * 1000 / CLOCK_SECOND;
}

static inline s64_t k_uptime_delta(s64_t *s)
{
    return k_uptime_get() - *s;
}

#define min(a, b) MIN(a, b)
#define max(a, b) MAX(a, b)

#else // CONTIKI

#include <memory.h>
#include <complex.h>

typedef float complex cplx;

static inline s64_t k_uptime_get(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static inline s64_t k_uptime_delta(s64_t *s)
{
    return k_uptime_get() - *s;
}

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

#define CONFIG_ARCH "native"
#define CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC 1

// square root
//#define arm_sqrt_f32 sqrtf

#endif
