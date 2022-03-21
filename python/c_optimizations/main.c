#include <stdio.h>
//#include "kiss_fft.h"
#include "threaded_distance.h"

void print_results(double* res, int length) {
  for (int i = 0; i < length; i++) {
      printf("%f ", res[i]);
  }
  printf("\n");
}

void dummy(double* in, int input_len, double* out, int output_len)
{
    for (int i = 0; i < input_len; i++) {
        out[i] = in[i];
    }
}

//int main() {
//
//    double buf1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
//    double buf2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
//
//    double* res1 = threaded_dist_calc((Dist_Func) &euclid_dist, &dummy, 5, buf1, buf2, 5, 10, 2);
//    double* res2 = threaded_dist_calc((Dist_Func) &cosine_dist, &dummy, 5, buf1, buf2, 5, 10, 2);
//    double* res3 = threaded_dist_calc((Dist_Func) &levenshtein_dist, &dummy, 5, buf1, buf2, 5, 10, 2);

//    print_results(res1, 10);
//    print_results(res2, 10);
//    print_results(res3, 10);
//}

void euclid_dist_shift(double* buf1, double* buf2, int input_len, int max_shift, double* result) {
    threaded_dist_calc((Dist_Func) &euclid_dist, 0, 0, buf1, buf2, input_len, max_shift, 5, result);
}

void cosine_dist_shift(double* buf1, double* buf2, int input_len, int max_shift, double* result) {
    threaded_dist_calc((Dist_Func) &cosine_dist, 0, 0, buf1, buf2, input_len, max_shift, 5, result);
}

void levenshtein_dist_shift(double* buf1, double* buf2, int input_len, int max_shift, double* result) {
    threaded_dist_calc((Dist_Func) &levenshtein_dist, 0, 0, buf1, buf2, input_len, max_shift, 5, result);
}

