#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "matrix.h"
#include "bit_gen.h"
#include "f2c.h"
#include "clapack.h"
#include "blaswrap.h"

int main() {

    float data[] = {2, 1, 0, 1, 1, 1, 0, 1, 2};

    float eigen_vector[3];

    // input
    uint32_t N = 3;             // order of the matrix
    uint32_t lda = 3;           // leading dimension of an array
    uint32_t upper_limit = 3;   // largest eigenvector to be returned
    uint32_t lower_limit = 3;   // smallest eigenvector to be returned
    uint32_t ldz = 3;           // leading dimension of eigen vector
    float work = 0;
    uint32_t lwork = -1;
    float absol = 0.0001;

    // output
    uint32_t eig_v_found = 0;      // eigenvectors found
    float eig_val = 0;             // eigenvalue found
    uint32_t iwork[3*5];
    uint32_t ifail[3];
    int info = 0;

    ssyevx_("Vectors", "Indices", "Upper", (integer*) &N, (real*) data, (integer*) &lda, (real*) 0, (real*) 0, (integer*) &lower_limit, (integer*) &upper_limit,
           (real*) &absol, (integer*) eig_v_found, (real*) &eig_val, (real*) eigen_vector, (integer*) ldz,  (integer*) &work, (integer*) &lwork, (integer*) &iwork,
           (integer*) &ifail, (integer*) info);

}
