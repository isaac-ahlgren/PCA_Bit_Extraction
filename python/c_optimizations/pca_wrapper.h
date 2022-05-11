#ifndef _PCA_WRAPPER_H
#define _PCA_WRAPPER_H

#include "headers.h"

struct fft_pca_args {
    struct fft_args* f_args;
    struct eig_decomp_args* e_args;
    uint32_t vec_len;
    uint32_t vec_num;
    float* fft_buf;
    float* cov_mat;
    float* cov_mat_means;
    float* eig_vec;
};

void fft_pca(float* input_buffer, float* output_buffer, void* args);
struct fft_pca_args* alloc_fft_pca_args(uint32_t vec_len, uint32_t vec_num);
void free_fft_pca_args(struct fft_pca_args* args);

#endif
