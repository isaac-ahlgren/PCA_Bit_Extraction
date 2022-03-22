#ifndef _FFT_WRAPPER
#define _FFT_WRAPPER

#include "kiss_fftr.h"

struct fft_args {
    kiss_fftr_cfg cfg;
    kiss_fft_cpx * tmp;
    int output_len;
};

void free_fft_args(struct fft_args* args);
struct fft_args* alloc_fft_args(int input_size);
void fft_abs(float* input, float* output, void* args);

#endif
