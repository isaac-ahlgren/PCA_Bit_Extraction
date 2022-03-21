#ifndef _THREADED_DIST_H
#define _THREADED_DIST_H

#include "distance.h"

typedef void (*Preprocess_Func)(double*, int, double*, int);
typedef double (*Dist_Func)(double*, double*, int);

void threaded_dist_calc(Dist_Func dist_func, Preprocess_Func pre_func, int preproc_output, double* buf1, double* buf2, int input_len, int max_shift, int thread_num, double* result);

#endif

