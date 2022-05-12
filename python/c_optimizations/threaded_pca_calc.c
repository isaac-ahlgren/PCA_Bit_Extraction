#include <pthread.h>
#include <stdio.h>
#include "threaded_pca_calc.h"

struct fft_calc_args {
    void* args;
    int input_len;
    float* buf;
    int output_len;
    float* out_buf;
    int buf_start_position;
    int out_start_position;
    int shift_len;
};

void* calc_fft_pca(void* input)
{
    struct fft_calc_args* inp = (struct fft_calc_args*) input;
    float* buf = &inp->buf[inp->buf_start_position];
    float* out_buf = &inp->out_buf[inp->out_start_position];

    for (int i = 0; i < inp->shift_len; i++) {
    	printf("%d\n", i);
       fft_pca(buf + i, out_buf + i*inp->output_len, inp->args);
    }
}

void threaded_calc_pca(float* buf, uint32_t vec_len, uint32_t vec_num, uint32_t max_shift, uint32_t thread_num, float* result)
{
    pthread_t* threads = malloc(sizeof(pthread_t)*thread_num);
    struct fft_calc_args* inputs = malloc(sizeof(struct fft_calc_args)*thread_num);

    uint32_t shift_len = max_shift / thread_num;

    for (int i = 0; i < thread_num; i++) {
        inputs[i].buf_start_position = i*shift_len;
        inputs[i].out_start_position = i*shift_len*vec_num;
        inputs[i].args = (void*) alloc_fft_pca_args(vec_len, vec_num);
        inputs[i].buf = buf;
        inputs[i].out_buf = result;
        inputs[i].output_len = vec_num;
        inputs[i].input_len = vec_len*vec_num;
        inputs[i].shift_len = shift_len;

        pthread_create(&threads[i], 0, &calc_fft_pca, (void*) &inputs[i]);
    }

    for (int i = 0; i < thread_num; i++) {
        pthread_join(threads[i], 0);
        free_fft_pca_args(inputs[i].args);
    }

    free(threads);
    free(inputs);
}
