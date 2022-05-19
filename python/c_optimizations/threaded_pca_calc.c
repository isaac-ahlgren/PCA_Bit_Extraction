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
    int max_shift;
};

struct eig_calc_args {
    struct fft_calc_args* args;
    float* eig_buffer;
    int* convergence_buffer;
    int conv_buffer_position;
    int eig_len;
    int eig_nums;
    int eig_buffer_position;
};

void* calc_fft_pca(void* input)
{
    struct eig_calc_args* eig_inp = (struct eig_calc_args*) input;
    struct fft_calc_args* inp = (struct fft_calc_args*) eig_inp->args;
    int* conv_buf = &eig_inp->convergence_buffer[eig_inp->conv_buffer_position];
    float* eig_buffer = &eig_inp->eig_buffer[eig_inp->eig_buffer_position];
    int eig_len = eig_inp->eig_len;
    int eig_nums = eig_inp->eig_nums;
    float* buf = &inp->buf[inp->buf_start_position];
    float* out_buf = &inp->out_buf[inp->out_start_position];

    for (int i = 0; i < inp->shift_len; i++) {
       fft_pca(buf + i, out_buf + i*inp->output_len, conv_buf + i*eig_nums*inp->shift_len, eig_buffer + i*eig_len*eig_nums, inp->args, inp->max_shift);
    }
}

void threaded_calc_pca(float* buf, uint32_t vec_len, uint32_t vec_num, uint32_t eig_vec_num, uint32_t max_shift, uint32_t thread_num, float* pca_samples, float* eigen_vectors, int* convergence)
{
    pthread_t* threads = malloc(sizeof(pthread_t)*thread_num);
    struct fft_calc_args* inputs = malloc(sizeof(struct fft_calc_args)*thread_num);
    struct eig_calc_args* eig_inp = malloc(sizeof(struct eig_calc_args)*thread_num);

    uint32_t shift_len = max_shift / thread_num;

    for (int i = 0; i < thread_num; i++) {
        inputs[i].buf_start_position = i*shift_len;
        inputs[i].out_start_position = i*shift_len*vec_num*eig_vec_num;
        inputs[i].args = (void*) alloc_fft_pca_args(vec_len, vec_num, eig_vec_num);
        inputs[i].buf = buf;
        inputs[i].out_buf = pca_samples;
        inputs[i].output_len = vec_num;
        inputs[i].input_len = vec_len*vec_num;
        inputs[i].shift_len = shift_len;
        inputs[i].max_shift = max_shift;

	eig_inp[i].args = &inputs[i];
        eig_inp[i].eig_len = vec_len/2 + 1;
        eig_inp[i].eig_buffer_position = i*shift_len*(vec_len/2 + 1)*eig_vec_num;
        eig_inp[i].eig_buffer = eigen_vectors;
	eig_inp[i].convergence_buffer = convergence;
	eig_inp[i].conv_buffer_position = i*shift_len

        pthread_create(&threads[i], 0, &calc_fft_pca, (void*) &eig_inp[i]);
    }

    for (int i = 0; i < thread_num; i++) {
        pthread_join(threads[i], 0);
        free_fft_pca_args(inputs[i].args);
    }

    free(threads);
    free(inputs);
    free(eig_inp);
}
