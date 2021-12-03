#include "matrix.h"

uint8_t mat_init_f64(matrix_instance_f64 * m,
                     uint16_t num_cols,
                     uint16_t num_rows,
                     double* data)
{
    m->num_cols = num_cols;
    m->num_rows = num_rows;
    m->data = data;
    return 1;
}  

uint8_t mat_mult_f64(const matrix_instance_f64 * src_a,
                     const matrix_instance_f64 * src_b,
                           matrix_instance_f64 * dst)
{
    uint16_t rows_a = src_a->num_rows;
    uint16_t columns_b = src_b->num_cols;
    uint16_t v_len = src_a->num_cols;
    
    for (uint16_t i = 0; i < columns_b; i++)
    {
        for (uint16_t j = 0; j < rows_a; j++)
        {
            uint16_t k = v_len;
            uint16_t pos_a = v_len * i;        /* position within the ith row vector of matrix A */
            uint16_t pos_b = j;                /* position within the jth column vector of matrix B */
            uint16_t pos_d = pos_a + pos_b;    /* postion (i,j) within the matrix of dst */
            dst->data[pos_d] = 0;
            while (k--) {
                dst->data[pos_d] += src_a->data[pos_a] * src_b->data[pos_b];
                pos_a++; pos_b += v_len;
            }
        }
    }

    dst->num_cols = rows_a;
    dst->num_rows = columns_b;

    return 1;
}

uint8_t mat_trans_f64(const matrix_instance_f64 * src,
		            matrix_instance_f64 * dest)
{
    uint16_t rows  = src->num_rows;
    uint16_t columns = src->num_cols;
    uint16_t v_len = rows;

    for (uint16_t i = 0; i < columns; i++)
    {
        for (uint16_t j = 0; j < rows; j++)
        {
            uint16_t pos_src = i * v_len + j;
            uint16_t pos_dest = j * v_len + i;
            dest->data[pos_dest] =  src->data[pos_src];
        }
    }

   dest->num_cols = src->num_rows;
   dest->num_rows = src->num_cols;

   return 1;
}
