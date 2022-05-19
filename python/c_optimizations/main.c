#include <stdio.h>
#include "pca_wrapper.h"
#include "threaded_pca_calc.h"

#define MAX_THREAD 1

void pca_shifted_calcs(float* buf, int vec_len, int vec_num, int eig_num, int max_shift, float* pca_samples, float* eig_vectors, int* convergence) {
    threaded_calc_pca(buf, vec_len, vec_num, eig_num, max_shift, MAX_THREAD, pca_samples, eig_vectors, convergence);
}
