#ifndef _THREADED_PCA_CALC_H
#define _THREADED_PCA_CALC_H

#include "headers.h"
#include "pca_wrapper.h"

void threaded_calc_pca(float* buf, uint32_t vec_len, uint32_t vec_num, uint32_t max_shift, uint32_t thread_num, float* result);
void threaded_calc_pca_eig(float* buf, uint32_t vec_len, uint32_t vec_num, uint32_t max_shift, uint32_t thread_num, float* convergence, float* eigen_vectors);

#endif

