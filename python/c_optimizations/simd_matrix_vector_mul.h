#ifndef _SIMD_MATRIX_VECTOR_MUL_H
#define _SIMD_MATRIX_VECTOR_MUL_H

void matrix_vec_mult_avx(float *mat, uint32_t dim_size, float *vec, float *new_vec);

#endif
