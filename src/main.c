#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "matrix.h"
#include "bit_gen.h"
#include "f2c.h"
#include "clapack.h"
#include "blaswrap.h"


int main() {

    float data[] = {1.5756, 1.4679, 0.4592, 1.4679, 1.5194, 0.7003, 0.4592, 0.7003, 0.9425};

    float eigen_vector[3];

    // input
    long int N = 3;             // order of the matrix
    long int lda = 3;           // leading dimension of an array
    long int upper_limit = 3;   // largest eigenvector to be returned
    long int lower_limit = 3;   // smallest eigenvector to be returned
    long int ldz = 3;           // leading dimension of eigen vector
    float* work = 0;
    long int lwork = -1;
    float absol = 0.0001;

    // output
    long int eig_v_found = 0;      // eigenvectors found
    float eig_val = 0;             // eigenvalue found
    long int iwork[3*5];
    long int ifail[3];
    long int info = 0;
    
    float workopt = 0;
    // allocate optimal workspace
    ssyevx_("Vectors", "Indices", "Upper", (integer*) &N, (real*) data, (integer*) &lda, (real*) 0, (real*) 0, (integer*) &lower_limit, (integer*) &upper_limit,
           (real*) &absol, (integer*) &eig_v_found, (real*) &eig_val, (real*) eigen_vector, (integer*) &ldz,  (real*) &workopt, (integer*) &lwork, (integer*) iwork,
           (integer*) ifail, (integer*) &info);

    lwork = (long int)workopt;
    work = (float*) malloc( lwork*sizeof(float));
    ssyevx_("Vectors", "Indices", "Upper", (integer*) &N, (real*) data, (integer*) &lda, (real*) 0, (real*) 0, (integer*) &lower_limit, (integer*) &upper_limit,
           (real*) &absol, (integer*) &eig_v_found, (real*) &eig_val, (real*) eigen_vector, (integer*) &ldz,  (real*) work, (integer*) &lwork, (integer*) iwork,
           (integer*) ifail, (integer*) &info);

    printf("%d\n", info);
    printf("%f %f %f\n", eigen_vector[0], eigen_vector[1], eigen_vector[2]);

}
