
#include <stdio.h>
#include "matrix.h"

int main() {

    matrix_instance_f32 m1 = {};
    matrix_instance_f32 m2 = {};
    matrix_instance_f32 m3 = {};
    float nums1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float nums2[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    mat_init_f32(&m1, 3, 3, nums1);
    mat_init_f32(&m2, 3, 3, nums2);

    mat_symm_f32(&m1, &m2);

    print_mat(&m1);
    printf("\n");
    print_mat(&m2);
}
