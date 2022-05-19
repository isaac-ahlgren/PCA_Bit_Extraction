#include <stdio.h>
#include "pca_wrapper.h"
#include "threaded_pca_calc.h"

#define MAX_THREAD 10

void pca_shifted_calcs(float* buf, int vec_len, int vec_num, int max_shift, float* result) {
    threaded_calc_pca(buf, vec_len, vec_num, max_shift, MAX_THREAD, result);
}
