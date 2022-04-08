#include <stdio.h>
#include "threaded_distance.h"
#include "pca_wrapper.h"

#define MAX_THREAD 10

void euclid_dist_shift(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc((Dist_Func) &euclid_dist, buf1, buf2, input_len, max_shift, MAX_THREAD, result);
}

void cosine_dist_shift(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc((Dist_Func) &cosine_dist, buf1, buf2, input_len, max_shift, MAX_THREAD, result);
}

void levenshtein_dist_shift(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc((Dist_Func) &levenshtein_dist, buf1, buf2, input_len, max_shift, MAX_THREAD, result);
}

void euclid_dist_shift_fft(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc_fft((Dist_Func) &euclid_dist, buf1, buf2, input_len, max_shift, MAX_THREAD, result);
}

void cosine_dist_shift_fft(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc_fft((Dist_Func) &cosine_dist, buf1, buf2, input_len, max_shift, MAX_THREAD, result);
}

void levenshtein_dist_shift_fft(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc_fft((Dist_Func) &levenshtein_dist, buf1, buf2, input_len, max_shift, MAX_THREAD, result);
}

void euclid_dist_shift_pca(float* buf1, float* buf2, int vec_len, int vec_num, int max_shift, float* result) {
    threaded_dist_calc_pca((Dist_Func) &euclid_dist, buf1, buf2, vec_len, vec_num,  max_shift, MAX_THREAD, result);
}

void cosine_dist_shift_pca(float* buf1, float* buf2, int vec_len, int vec_num, int max_shift, float* result) {
    threaded_dist_calc_pca((Dist_Func) &cosine_dist, buf1, buf2, vec_len, vec_num,  max_shift, MAX_THREAD, result);
}

void levenshtein_dist_shift_pca(float* buf1, float* buf2, int vec_len, int vec_num, int max_shift, float* result) {
    threaded_dist_calc_pca((Dist_Func) &levenshtein_dist, buf1, buf2, vec_len, vec_num,  max_shift, MAX_THREAD, result);
}

void gen_pca_samples(float* input_buffer, float* output_buffer, int vec_len, int vec_num, int starting_pos)
{
    struct fft_pca_args* args = alloc_fft_pca_args(vec_len, vec_num);
    float* data = input_buffer + starting_pos;
    fft_pca(data, output_buffer, (void*) args);
    free_fft_pca_args(args);
}

