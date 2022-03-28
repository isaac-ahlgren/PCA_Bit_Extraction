#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
//#include "pca_wrapper.h"
#include "eig_sym.h"
#include "fft_wrapper.h"

float mean(float* vector, int vec_len, int vec_num)
{
    float sum = 0;
    for (int i = 0; i < vec_num; i++)
    {
        sum += vector[i*vec_len]; // because matrices are row major instead of column major
    }

    return sum / vec_num;
}

void calc_means(float* A, float* means, int vec_len, int vec_num)
{
    for (int i = 0; i < vec_len; i++)
    {
        means[i] = 0;
        means[i] = mean(&A[i], vec_len, vec_num);
    }
}   
        
void cov(float* A, float* cov_mat, float* means, int vec_len, int vec_num)
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

void print_matrix(float* mat, int columns, int rows) {
    for (int i = 0; i < columns*rows; i++) {
        printf("%f ", mat[i]);
        if ((i + 1) % columns == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

struct fft_pca_args { 
    struct fft_args* f_args;
    int vec_len;
    int vec_num;
    float* cov_mat;
    float* cov_mat_means;
    float* eig_vals;
    float* princ_comp;
};

struct fft_pca_args* alloc_fft_pca_args(int input_size, int vec_len, int vec_num)
{
    struct fft_pca_args* args = malloc(sizeof(struct fft_pca_args));
    
    args->f_args = alloc_fft_args(vec_len);
    args->vec_len = vec_len;
    args->vec_num = vec_num;
    args->cov_mat = malloc(sizeof(float)*vec_len*vec_len);
    args->cov_mat_means = malloc(sizeof(float)*vec_len);
    args->eig_vals = malloc(sizeof(float)*vec_len);
    args->princ_comp = malloc(sizeof(float)*vec_len);
}

void free_fft_pca_args(struct fft_pca_args* args)
{
    free_fft_args(args->f_args);
    free(args->cov_mat);
    free(args->cov_mat_means);
    free(args->eig_vals);
    free(args->princ_comp);
    free(args);
}

void fft_obs_matrix(float* buffer, int vec_len, int vec_num, struct fft_args* args)
{
    for (int i = 0; i < vec_num; i++)
    {
        /*
            Since the data is real, the output should be length n/2 + 1. So,
            placing the output as (buffer + i*(vec_len/2 + 1)) allows us to 
            condense the matrix without leaving any extra space because of
            how real ffts work.
        */ 
        fft_abs((buffer + i*vec_len), (buffer + i*(vec_len/2 + 1)), args);
    } 
}

void find_princ_comp(float* conv_mat, float* eig_vals, float* princ_comp, int vec_len)
{

    int max = -1;
    for (int i = 0; i < vec_len; i++)
    {
        if (max < 0 || eig_vals[max] > eig_vals[i]) {
            max = i;
        }
    }

    for (int i = 0; i < vec_len; i++)
    {
        princ_comp[i] = conv_mat[max + i*vec_len];
    }
}

void project_data(float* buffer, float* princ_comp, float* output, int vec_len, int vec_num)
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
    float* cov_mat = inputs->cov_mat;
    float* cov_mat_means = inputs->cov_mat_means;
    float* eig_vals = inputs->eig_vals;
    float* princ_comp = inputs->princ_comp;
    int vec_len = inputs->vec_len;
    int vec_num = inputs->vec_num;

    print_matrix(input_buffer, vec_len, vec_num);
    printf("\n");

    fft_obs_matrix(input_buffer, vec_len, vec_num, inputs->f_args);

    vec_len = (vec_len/2 + 1); // since data is real, vectors after fft are length n/2 + 1 

    print_matrix(input_buffer, vec_len, vec_num);
    printf("\n");
    
    cov(input_buffer, cov_mat, cov_mat_means, vec_len, vec_num);

    print_matrix(cov_mat, vec_len, vec_len);
    printf("\n");

    eig_sym(cov_mat, vec_len, eig_vals);

    print_matrix(cov_mat, vec_len, vec_len);
    printf("\n");

    find_princ_comp(cov_mat, eig_vals, princ_comp, vec_len);

    print_matrix(princ_comp, vec_len, 1);
    printf("\n");

    project_data(input_buffer, princ_comp, output_buffer, vec_len, vec_num);

    print_matrix(output_buffer, vec_num, 1);
    printf("\n");
}

int main() {
    float data[] = {1, 2, 6, 4, 3, 9, 7, 0, 12, 4, 7, 23, 9, 2, 7 , 3, 1, 3, 7, 1, 2, 9, 2, 1, 4, 5, 3, 1, 8, 2, 1, 1};
    float output[] = {0, 0};

    struct fft_pca_args* args = alloc_fft_pca_args(32, 16, 2);
    fft_pca(data, output, (void*) args);
}
