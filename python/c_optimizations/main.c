#include <stdio.h>
#include "threaded_distance.h"

#define MAX_THREAD 16

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
