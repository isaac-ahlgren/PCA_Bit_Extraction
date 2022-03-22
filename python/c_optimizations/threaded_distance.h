#ifndef _THREADED_DIST_H
#define _THREADED_DIST_H

#include "distance.h"

typedef float (*Dist_Func)(float*, float*, int);

void threaded_dist_calc(Dist_Func dist_func, float* buf1, float* buf2, int input_len, int max_shift, int thread_num, float* result);
void threaded_dist_calc_fft(Dist_Func dist_func, float* buf1, float* buf2, int input_len, int max_shift, int thread_num, float* result);

#endif

