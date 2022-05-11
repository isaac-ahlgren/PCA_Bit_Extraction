#include <stdio.h>
#include "threaded_distance.h"
#include "pca_wrapper.h"
#include "threaded_pca_calc.h"

#define MAX_THREAD 1

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

void pca_shifted_calcs(float* buf, int vec_len, int vec_num, int max_shift, float* result) {
    threaded_calc_pca(buf, vec_len, vec_num, max_shift, MAX_THREAD, result);
}
/*
float buf[8192*64 + 5000] = {1};
float res[64*5000];

int main()
{
    pca_shifted_calcs(buf, 8192, 64, 5000, res);
}
*/
