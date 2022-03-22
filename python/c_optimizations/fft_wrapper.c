#include <math.h>
#include <stdlib.h>
#include "fft_wrapper.h"

struct fft_args* alloc_fft_args(int input_size) 
{
    struct fft_args* args = malloc(sizeof(struct fft_args));
    
    args->cfg = kiss_fftr_alloc(input_size, 0, 0, 0);
    args->tmp = malloc(sizeof(kiss_fft_cpx)*(input_size/2 + 1));
    args->output_len = input_size/2 + 1;

    return args;
}

void free_fft_args(struct fft_args* args) 
{
     free(args->tmp);
     kiss_fftr_free(args->cfg);
     free(args);
}

void magnitude(kiss_fft_cpx* in, float* out, int length)
{
    for (int i = 0; i < length; i++) {
        out[i] = sqrt(in[i].r*in[i].r + in[i].i*in[i].i);
    }
}

void fft_abs(float* input, float* output, void* args) 
{
    struct fft_args* inputs = (struct fft_args*) args;

    kiss_fftr(inputs->cfg, (const kiss_fft_scalar*) input, inputs->tmp);
    
    magnitude(inputs->tmp, output, inputs->output_len);
}
