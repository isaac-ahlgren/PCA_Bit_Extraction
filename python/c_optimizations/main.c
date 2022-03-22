#include <stdio.h>
#include "threaded_distance.h"

void euclid_dist_shift(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc((Dist_Func) &euclid_dist, buf1, buf2, input_len, max_shift, 8, result);
}

void cosine_dist_shift(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc((Dist_Func) &cosine_dist, buf1, buf2, input_len, max_shift, 8, result);
}

void levenshtein_dist_shift(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc((Dist_Func) &levenshtein_dist, buf1, buf2, input_len, max_shift, 8, result);
}

void euclid_dist_shift_fft(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc_fft((Dist_Func) &euclid_dist, buf1, buf2, input_len, max_shift, 8, result);
}

void cosine_dist_shift_fft(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc_fft((Dist_Func) &cosine_dist, buf1, buf2, input_len, max_shift, 8, result);
}

void levenshtein_dist_shift_fft(float* buf1, float* buf2, int input_len, int max_shift, float* result) {
    threaded_dist_calc_fft((Dist_Func) &levenshtein_dist, buf1, buf2, input_len, max_shift, 8, result);
}
