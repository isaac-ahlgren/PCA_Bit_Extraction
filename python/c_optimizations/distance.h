#ifndef _DIST_H
#define _DIST_H

float euclid_dist(float* x, float* y, int length);
float cosine_dist(float* x, float* y, int length);
int levenshtein_dist(float* x, float* y, int length);

#endif
