#include <pthread.h>
#include <stdlib.h>
#include "threaded_distance.h"
#include "fft_wrapper.h"

typedef void (*Preprocess_Func)(float*, float*, void*);

struct args {
    Dist_Func dist_func;
    Preprocess_Func pre_func;
    void* preprocess_args;
    int input_len;
    float* buf1;
    float* buf2;
    int preproc_output;
    float* out_buf1;
    float* out_buf2;
    int start_position;
    int shift_len;
    float* results;
};

void* calc_dist(void* input)
{
    struct args* inp = (struct args*) input;
    float* buffer1 = inp->buf1;
    float* buffer2 = inp->buf2;
    int preproc_flag = (inp->pre_func != 0);
    int dist_func_len = inp->input_len;

    if (preproc_flag) {
        buffer1 = inp->out_buf1;
        dist_func_len = inp->preproc_output;

        inp->pre_func(inp->buf1, inp->out_buf1, inp->preprocess_args);
    }

    for (int i = inp->start_position; i < (inp->start_position + inp->shift_len); i++) {

        if (preproc_flag) {
            inp->pre_func(inp->buf2 + i, inp->out_buf2, inp->preprocess_args);
            buffer2 = inp->out_buf2;
        }
        else {
            buffer2 = inp->buf2 + i;
        }

        inp->results[i] = inp->dist_func(buffer1, buffer2, dist_func_len);
    }
    return 0;
}

void threaded_dist_calc(Dist_Func dist_func, float* buf1, float* buf2, int input_len, int max_shift, int thread_num, float* result)
{
    pthread_t* threads = malloc(sizeof(pthread_t)*thread_num);
    struct args* inputs = malloc(sizeof(struct args)*thread_num);

    int shift_len = max_shift / thread_num;

    for (int i = 0; i < thread_num; i++) {
        inputs[i].start_position = i*shift_len;
        inputs[i].dist_func = dist_func;
        inputs[i].pre_func = 0;
        inputs[i].preprocess_args = 0;
        inputs[i].buf1 = buf1;
        inputs[i].buf2 = buf2;
        inputs[i].input_len = input_len;
        inputs[i].results = result;
        inputs[i].shift_len = shift_len;

        pthread_create(&threads[i], 0, calc_dist, (void*) &inputs[i]);
    }

    for (int i = 0; i < thread_num; i++) {
        pthread_join(threads[i], 0);
    }
    
    free(threads);
    free(inputs);
}

void threaded_dist_calc_fft(Dist_Func dist_func, float* buf1, float* buf2, int input_len, int max_shift, int thread_num, float* result)
{
    pthread_t* threads = malloc(sizeof(pthread_t)*thread_num);
    struct args* inputs = malloc(sizeof(struct args)*thread_num);

    int shift_len = max_shift / thread_num;

    for (int i = 0; i < thread_num; i++) {
        inputs[i].start_position = i*shift_len;
        inputs[i].dist_func = dist_func;
        inputs[i].pre_func = &fft_abs;
        inputs[i].preprocess_args = (void*) alloc_fft_args(input_len);
        inputs[i].buf1 = buf1;
        inputs[i].buf2 = buf2;
        inputs[i].input_len = input_len;
        inputs[i].results = result;
        inputs[i].shift_len = shift_len;

        pthread_create(&threads[i], 0, calc_dist, (void*) &inputs[i]);
    }

    for (int i = 0; i < thread_num; i++) {
        pthread_join(threads[i], 0);
        free_fft_args(inputs[i].preprocess_args);
    }
    
    free(threads);
    free(inputs);
}
