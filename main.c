#include <stdio.h>
#include "matrix.h"

int print_mat(matrix_instance_f64 * m)
{
    uint16_t buffer_size = m->num_rows * m->num_cols;
    for (int i = 0; i < buffer_size; i++)
    {
        printf("%f", m->data[i]);
        if ((i + 1) % m->num_cols > 0)
        {
            printf(", ");
        }
        else
        {
            printf("\n");
        }
    }
}

int main() {

    matrix_instance_f64 m1 = {};
    matrix_instance_f64 m2 = {};
    matrix_instance_f64 m3 = {};
    double nums1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double nums2[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double nums3[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    mat_init_f64(&m1, 3, 3, nums1);
    mat_init_f64(&m2, 3, 3, nums2);
    mat_init_f64(&m3, 3, 3, nums3);

    mat_trans_f64(&m1, &m2);

    mat_mult_f64(&m1, &m2, &m3);

    print_mat(&m1);
    printf("\n");
    print_mat(&m2);
    printf("\n");
    print_mat(&m3);
}
