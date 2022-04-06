#ifndef _THREADED_DIST_H
#define _THREADED_DIST_H

#include "headers.h"
#include "distance.h"

typedef float (*Dist_Func)(float*, float*, int);

void threaded_dist_calc(Dist_Func dist_func, float* buf1, float* buf2, uint32_t input_len, uint32_t max_shift, uint32_t thread_num, float* result);
void threaded_dist_calc_fft(Dist_Func dist_func, float* buf1, float* buf2, uint32_t input_len, uint32_t max_shift, uint32_t thread_num, float* result);
void threaded_dist_calc_pca(Dist_Func dist_func, float* buf1, float* buf2, uint32_t vec_len, uint32_t vec_num,  uint32_t max_shift, uint32_t thread_num, float* result);

#endif

