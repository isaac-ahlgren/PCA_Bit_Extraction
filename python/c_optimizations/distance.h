#ifndef _DIST_H
#define _DIST_H

#include "headers.h"

float euclid_dist(float* x, float* y, uint32_t length);
float cosine_dist(float* x, float* y, uint32_t length);
int levenshtein_dist(float* x, float* y, uint32_t length);

#endif
