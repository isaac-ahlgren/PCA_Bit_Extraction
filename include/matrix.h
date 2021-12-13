#ifndef _MATRIX_H
#define _MATRIX_H

#include <stdint.h>

/*
    Matrix structure that is indexed into using the following formula for
    the entry A(i,j).

    A[i + rows*j] = A(i,j)
*/

typedef struct
{
    uint16_t num_rows;
    uint16_t num_cols;
    float *data;
} matrix_instance_f32;

uint8_t mat_init_f32(matrix_instance_f32 * m,
                     uint16_t num_cols,
                     uint16_t num_rows,
                     float* data);

uint8_t print_mat(matrix_instance_f32 * m);

uint8_t mat_symm_f32(const matrix_instance_f32 * src,
                           matrix_instance_f32 * dst);

#endif

