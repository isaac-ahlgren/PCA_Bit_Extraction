#ifndef EIG_VEC_DECOMP_H
#define EIG_VEC_DECOMP_H

struct eig_decomp_args {
    float*   s;         // mem needed to calc error
    uint32_t dim_size;
    float*   eig_vec;
    uint32_t execs;    // maxmimum loop executions before giving up on convergence
    float    err_tol;  // error tolerated in the eigenvector
};

void free_eig_args(struct eig_decomp_args* args);
struct eig_decomp_args* alloc_eig_args(uint32_t dim_size, uint32_t execs, float err_tol);
void eig_decomp(float* matrix, struct eig_decomp_args* args);

#endif
