#include <stdio.h>
#include "matrix.h"

uint8_t print_mat(matrix_instance_f32 * m)
{
    int rows = m->num_rows;
    int columns = m->num_cols;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
           printf("%f ", m->data[i + j*rows]);
        }
        printf("\n");
    }
}

uint8_t mat_init_f32(matrix_instance_f32 * m,
                     uint16_t num_cols,
                     uint16_t num_rows,
                     float* data)
{
    m->num_cols = num_cols;
    m->num_rows = num_rows;
    m->data = data;
    return 1;
}

/*
    Function that symmetrizes a matrix using A^T*A. Computes only the upper triangular
    portion of the matrix.
*/
uint8_t mat_symm_f32(const matrix_instance_f32 * src,
                           matrix_instance_f32 * dst)
{
    uint16_t rows = src->num_rows;
    uint16_t columns = src->num_cols;
    uint16_t vlen = src->num_rows;

    for  (uint16_t i = 0; i < rows; i++)
    {
        for (uint16_t j = i; j < columns; j++)
        {
            uint16_t k = vlen;
            uint16_t pos_a = vlen * j;         /* beginning position ith row vector */
            uint16_t pos_b = i;                /* beginning position of jth column vector */
            uint16_t pos_d = pos_a + pos_b;    /* postion (i,j) within the matrix of dst */
            dst->data[pos_d] = 0;
            while (k--) {
                dst->data[pos_d] += src->data[pos_a] * src->data[pos_b];
                pos_a += vlen; pos_b++;
            }
        }
    }

    dst->num_cols = columns;
    dst->num_rows = columns;

    return 1;
}
