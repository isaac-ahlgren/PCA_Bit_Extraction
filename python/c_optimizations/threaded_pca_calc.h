#ifndef _THREADED_PCA_CALC_H
#define _THREADED_PCA_CALC_H

#include "headers.h"
#include "pca_wrapper.h"

void threaded_calc_pca(float* buf, uint32_t vec_len, uint32_t vec_num, uint32_t eig_vec_num, uint32_t max_shift, uint32_t thread_num, float* pca_samples, float* eigen_vectors, int* convergence);

#endif

