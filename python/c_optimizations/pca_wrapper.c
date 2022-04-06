#include "pca_wrapper.h"
#include "f2c.h"
#include "clapack.h"
#include "fft_wrapper.h"

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

void print_matrix(float* mat, int columns, int rows) {
    for (int i = 0; i < columns*rows; i++) {
        printf("%f ", mat[i]);
        if ((i + 1) % columns == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

struct fft_pca_args* alloc_fft_pca_args(uint32_t vec_len, uint32_t vec_num)
{
    struct fft_pca_args* args = malloc(sizeof(struct fft_pca_args));
    
    args->f_args = alloc_fft_args(vec_len);
    args->vec_len = vec_len;
    args->vec_num = vec_num;
    args->fft_buf = malloc(sizeof(float)*(vec_len/2 + 1)*vec_num);
    args->cov_mat = malloc(sizeof(float)*vec_len*vec_len);
    args->cov_mat_means = malloc(sizeof(float)*vec_len);
    args->princ_comp = malloc(sizeof(float)*vec_len);
    args->iwork = malloc(sizeof(uint32_t)*vec_len*5);


    integer n = vec_len, lda = vec_len, ldz = vec_len;

    // Allocate correct amount of memory for work
    integer lower_limit = vec_len;
    integer upper_limit = vec_len;
    integer eig_val_found = 0;
    integer ifail = 0;
    real work = 0;
    integer info = 0;
    real eig_val = 0;
    real absol = 0.001;
    integer lwork = -1;
    ssyevx_("Vectors", "Indices", "Upper", (integer*) &n, (real*) args->cov_mat, (integer*) &lda, (real*) 0, (real*) 0, (integer*) &lower_limit, (integer*) &upper_limit,
           (real*) &absol, (integer*) &eig_val_found, (real*) &eig_val, (real*) args->princ_comp, (integer*) &ldz, (real*) &work, (integer*) &lwork, (integer*) args->iwork,
           (integer*) &ifail, (integer*) &info);
    
    args->lwork = 10000;
    args->work = malloc(sizeof(float)*10000);

    return args;
}

void free_fft_pca_args(struct fft_pca_args* args)
{
    free_fft_args(args->f_args);
    free(args->fft_buf);
    free(args->cov_mat);
    free(args->cov_mat_means);
    free(args->princ_comp);
    free(args->iwork);
    free(args->work);
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
    float* princ_comp = inputs->princ_comp;
    float* work = inputs->work;
    uint32_t* iwork = inputs->iwork;
    uint32_t lwork = inputs->lwork;
    uint32_t vec_len = inputs->vec_len;
    uint32_t vec_num = inputs->vec_num;

    fft_obs_matrix(input_buffer, fft_buf,  vec_len, vec_num, inputs->f_args);

    vec_len = (vec_len/2 + 1); // since data is real, vectors after fft are length n/2 + 1 
    
    cov(fft_buf, cov_mat, cov_mat_means, vec_len, vec_num);

    integer n = vec_len, lda = vec_len, ldz = vec_len;
    integer lower_limit = vec_len;
    integer upper_limit = vec_len;
    integer eig_val_found = 0;
    integer ifail = 0;
    integer info = 0;
    real eig_val = 0;
    real absol = 0.001;
    ssyevx_("Vectors", "Indices", "Upper", (integer*) &n, (real*) cov_mat, (integer*) &lda, (real*) 0, (real*) 0, (integer*) &upper_limit, (integer*) &lower_limit,
           (real*) &absol, (integer*) &eig_val_found, (real*) &eig_val, (real*) princ_comp, (integer*) &ldz, (real*) work, (integer*) &lwork, (integer*) iwork,
           (integer*) &ifail, (integer*) &info);

    project_data(fft_buf, princ_comp, output_buffer, vec_len, vec_num);

    fix_output(output_buffer, vec_num);
}

/*
int main() {
    float data[] = {1, 2, 6, 4, 3, 9, 7, 0, 12, 4, 7, 23, 9, 2, 7 , 3, 1, 3, 7, 1, 2, 9, 2, 1, 4, 5, 3, 1, 8, 2, 1, 1};
    float output[] = {0, 0};

    struct fft_pca_args* args = alloc_fft_pca_args(32, 16, 2);
    fft_pca(data, output, (void*) args);
    free_fft_pca_args(args);
    
}  
*/
