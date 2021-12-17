#include <stdint.h>
#include <stdlib.h>
#include "matrix.h"
#include "f2c.h"
#include "clapack.h"

uint8_t time_resistant_bit_extraction(float* data,
                                      uint16_t data_length
                                      uint16_t vector_nums
                                      float absol)
{
    uint32_t vlen = data_length / vector_nums;

    // initialize matrix for data matrix
    matrix_instance_f32 data_matrix = {};
    mat_init_f32(&data_matrix, vlen, vector_nums, data);

    // initialze matrix for covariance matrix
    float* cov_data = (float*) malloc(vlen*vlen*sizeof(float));
    mat_init_f32(&cov_matrix, vlen, vlen, cov_space);

    mat_symm_f32(&data_matrix, &cov_matrix);

    // initialize memory for eigenvector
    float* eigen_vector = (float*) malloc(vlen*sizeof(float));

    // input
    char job = 'V';                // calculate eigenvectors and eigen values
    char range = 'I";              // selective eigenvectors and eigenvalues will be found
    char uplo = 'U';               // upper triangular of symmetric matrix is stored 
    uint32_t N = vlen;             // order of the matrix
    uint32_t lda = vlen;           // leading dimension of an array
    uint32_t upper_limit = vlen;   // largest eigenvector to be returned
    uint32_t lower_limit = vlen;   // smallest eigenvector to be returned
    uint32_t ldz = vlen;           // leading dimension of eigen vector
    uint32_t work = 0;     
    uint32_t lwork = -1;

    // output
    uint32_t eig_v_found = 0;      // eigenvectors found
    float eig_val = 0;             // eigenvalue found
    uint32_t iwork = 0;
    uint32_t ifail = 0;
    int info = 0;

    ssyev_(&job, &range, &uplo, (integer*) &N, (real*) cov_matrix.data, (integer*) &lda, (real*) 0, (real*) 0, (integer*) &lower_limit, (integer*) &upper_limit, 
           (real*) &absol, (integer*) eig_v_found, (real*) &eig_val, (real*) eig_vector, (integer*) ldz,  (integer*) &work, (integer*) &lwork, (integer*) &iwork,
           (integer*) &ifail, (integer*) info);

    // free allocated memory
    free(eigen_vector);
    free(cov_data);
}

