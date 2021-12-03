#ifndef _MATRIX_H
#define _MATRIX_H

#include <stdint.h>

typedef struct
{
    uint16_t num_rows;
    uint16_t num_cols;
    double *data;
} matrix_instance_f64;

uint8_t mat_init_f64(matrix_instance_f64 * m,
                     uint16_t num_cols,
                     uint16_t num_rows,
                     double* data);

uint8_t mat_mult_f64(
    const matrix_instance_f64 * src_a,
    const matrix_instance_f64 * src_b,
          matrix_instance_f64 * dst);

uint8_t mat_trans_f64(
    const matrix_instance_f64 * src,
          matrix_instance_f64 * dst);

#endif
