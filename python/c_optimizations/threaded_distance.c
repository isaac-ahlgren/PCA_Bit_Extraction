#include <pthread.h>
#include <stdlib.h>
#include "threaded_distance.h"

struct args {
    Dist_Func dist_func;
    Preprocess_Func pre_func;
    int input_len;
    double* buf1;
    double* buf2;
    int preproc_output;
    double* out_buf1;
    double* out_buf2;
    int start_position;
    int shift_len;
    double* results;
};

void* calc_dist(void* input)
{
    struct args* inp = (struct args*) input;
    double* buffer1 = inp->buf1;
    double* buffer2 = inp->buf2;
    int preproc_flag = (inp->pre_func != 0);
    int dist_func_len = inp->input_len;

    if (preproc_flag) {
        buffer1 = inp->out_buf1;
        dist_func_len = inp->preproc_output;

        inp->pre_func(inp->buf1, inp->input_len, inp->out_buf1, inp->preproc_output);
    }

    for (int i = inp->start_position; i < (inp->start_position + inp->shift_len); i++) {

        if (preproc_flag) {
            inp->pre_func(inp->buf2 + i, inp->input_len, inp->out_buf2, inp->preproc_output);
            buffer2 = inp->out_buf2;
        }
        else {
            buffer2 = inp->buf2 + i;
        }

        inp->results[i] = inp->dist_func(buffer1, buffer2, dist_func_len);
    }
    return 0;
}

void threaded_dist_calc(Dist_Func dist_func, Preprocess_Func pre_func, int preproc_output, double* buf1, double* buf2, int input_len, int max_shift, int thread_num, double* result)
{
    pthread_t* threads = malloc(sizeof(pthread_t)*thread_num);
    struct args* inputs = malloc(sizeof(struct args)*thread_num);

    int shift_len = max_shift / thread_num;

    for (int i = 0; i < thread_num; i++) {
        inputs[i].start_position = i*shift_len;
        inputs[i].dist_func = dist_func;
        inputs[i].pre_func = pre_func;
        inputs[i].buf1 = buf1;
        inputs[i].buf2 = buf2;
        inputs[i].input_len = input_len;
        inputs[i].results = result;
        inputs[i].shift_len = shift_len;

        // Allocate memory for output of preprocess function
        if (inputs[i].pre_func != 0) {
            inputs[i].out_buf1 = malloc(sizeof(double)*preproc_output);
            inputs[i].out_buf2 = malloc(sizeof(double)*preproc_output);
            inputs[i].preproc_output = preproc_output;
        }

        pthread_create(&threads[i], 0, calc_dist, (void*) &inputs[i]);
    }

    for (int i = 0; i < thread_num; i++) {
        pthread_join(threads[i], 0);
        
        if (inputs[i].pre_func != 0) {
            free(inputs[i].out_buf1);
            free(inputs[i].out_buf2);
        }
    }
    
    free(threads);
    free(inputs);
}


