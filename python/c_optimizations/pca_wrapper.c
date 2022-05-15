#include "pca_wrapper.h"
#include "eig_vec_decomp.h"
#include "fft_wrapper.h"
#include <stdio.h>

float mean(float* vector, uint32_t vec_len, uint32_t vec_num)
{
    float sum = 0;
    for (int i = 0; i < vec_num; i++)
    {
        sum += vector[i*vec_len]; // because matrices are row major instead of column major
    }

    return sum / vec_num;
}

void calc_means(float* A, float* means, uint32_t vec_len, uint32_t vec_num)
{
    for (int i = 0; i < vec_len; i++)
    {
        means[i] = 0;
        means[i] = mean(&A[i], vec_len, vec_num);
    }
}   
        
void cov(float* A, float* cov_mat, float* means, uint32_t vec_len, uint32_t vec_num)
{
    
    calc_means(A, means, vec_len, vec_num);

    for (int i = 0; i < vec_len; i++) {
        
        for (int j = i; j < vec_len; j++) {
            // calculate covariance between two vectors
            cov_mat[i*vec_len + j] = 0;
            for (int k = 0; k < vec_num; k++) {
                cov_mat[i*vec_len + j] += (A[k*vec_len + i] - means[i]) * (A[k*vec_len + j] - means[j]);
            }
            cov_mat[i*vec_len + j] /= vec_num - 1;
            if (i != j) {
                cov_mat[j*vec_len + i] = cov_mat[i*vec_len + j];
            }
        }
    }
}

/*
void print_matrix(float* mat, int columns, int rows) {
    for (int i = 0; i < columns*rows; i++) {
        printf("%f ", mat[i]);
        if ((i + 1) % columns == 0) {
            printf("\n");
        }
    }
    printf("\n");
}
*/
struct fft_pca_args* alloc_fft_pca_args(uint32_t vec_len, uint32_t vec_num)
{
    struct fft_pca_args* args = malloc(sizeof(struct fft_pca_args));
    
    args->f_args = alloc_fft_args(vec_len);
    args->e_args = alloc_eig_args((vec_len/2 + 1), 10000, 0.01);
    args->vec_len = vec_len;
    args->vec_num = vec_num;
    args->fft_buf = malloc(sizeof(float)*(vec_len/2 + 1)*vec_num);
    args->cov_mat = malloc(sizeof(float)*(vec_len/2 + 1)*(vec_len/2 + 1));
    args->cov_mat_means = malloc(sizeof(float)*(vec_len/2 + 1));
    args->eig_vec = malloc(sizeof(float)*(vec_len/2 + 1));

    return args;
}

void free_fft_pca_args(struct fft_pca_args* args)
{
    free_fft_args(args->f_args);
    free_eig_args(args->e_args);
    free(args->fft_buf);
    free(args->cov_mat);
    free(args->cov_mat_means);
    free(args->eig_vec);
    free(args);
}

void fft_obs_matrix(float* input, float* output, uint32_t vec_len, uint32_t vec_num, struct fft_args* args)
{
    for (int i = 0; i < vec_num; i++)
    {
        /*
            Since the data is real, the output should be length n/2 + 1. So,
            placing the output as (buffer + i*(vec_len/2 + 1)) allows us to 
            condense the matrix without leaving any extra space because of
            how real ffts work.
        */ 
        fft_abs((input + i*vec_len), (output + i*(vec_len/2 + 1)), args);
    } 
}

void fix_output(float* output, uint32_t output_len)
{
    for (int i = 0; i < output_len; i++)
    {
        output[i] = (output[i] < 0) ? output[i]*-1 : output[i];
    }  
}

void project_data(float* buffer, float* princ_comp, float* output, uint32_t vec_len, uint32_t vec_num)
{
    for (int i = 0; i < vec_num; i++)
    {
        output[i] = 0;
        for (int j = 0; j < vec_len; j++)
        {
            output[i] += buffer[i*vec_len + j] * princ_comp[j];
        }
    }
}

void fft_pca(float* input_buffer, float* output_buffer, void* args)
{
    struct fft_pca_args* inputs = (struct fft_pca_args*) args;
    float* fft_buf = inputs->fft_buf;
    float* cov_mat = inputs->cov_mat;
    float* cov_mat_means = inputs->cov_mat_means;
    uint32_t vec_len = inputs->vec_len;
    uint32_t vec_num = inputs->vec_num;

    fft_obs_matrix(input_buffer, fft_buf,  vec_len, vec_num, inputs->f_args);

    vec_len = (vec_len/2 + 1); // since data is real, vectors after fft are length n/2 + 1 
    
    cov(fft_buf, cov_mat, cov_mat_means, vec_len, vec_num);
   
    eig_decomp(cov_mat, inputs->e_args);

    float* eig_vec = inputs->e_args->eig_vec;

    fix_output(eig_vec, vec_len);

    project_data(fft_buf, eig_vec, output_buffer, vec_len, vec_num);
}

void fft_pca_eig(float* input_buffer, float* convergence, float* eigen_vectors, void* args)
{
    struct fft_pca_args* inputs = (struct fft_pca_args*) args;
    float* fft_buf = inputs->fft_buf;
    float* cov_mat = inputs->cov_mat;
    float* cov_mat_means = inputs->cov_mat_means;
    uint32_t vec_len = inputs->vec_len;
    uint32_t vec_num = inputs->vec_num;

    fft_obs_matrix(input_buffer, fft_buf,  vec_len, vec_num, inputs->f_args);

    vec_len = (vec_len/2 + 1); // since data is real, vectors after fft are length n/2 + 1 

    cov(fft_buf, cov_mat, cov_mat_means, vec_len, vec_num);

    *convergence = eig_decomp(cov_mat, inputs->e_args);

    float* eig_vec = inputs->e_args->eig_vec;

    fix_output(eig_vec, vec_len);

    for (int i = 0; i < vec_len; i++) {
        eigen_vectors[i] = eig_vec[i];
    }
}
