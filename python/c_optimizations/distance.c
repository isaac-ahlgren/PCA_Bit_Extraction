#include <math.h>
#include <stdlib.h>

double euclid_dist(double* x, double* y, int length)
{
    double total = 0;
    for (int i = 0; i < length; i++) {
        total += pow(x[i] - y[i], 2);
    }

    return sqrt(total);
}

double cosine_dist(double* x, double* y, int length)
{
    double total = 0;
    for (int i = 0; i < length; i++) {
        total += x[i]*y[i];
    }

    return total;
}

int levenshtein_dist(double* x, double* y, int length)
{
    double* distances = malloc(sizeof(double)*(length+1)*(length+1));

    for (int i = 0; i < length + 1; i++) {
        distances[i] = i;
        distances[i*(length+1)] = i;
    }

    for (int i = 1; i < length + 1; i++) {

        for (int j = 1; j < length + 1; j++) {
            
            if (x[i-1] == y[j-1]) {
                distances[i + j*(length+1)] = distances[(i - 1) + (j - 1)*(length+1)];
            }
            else {
                int dist_a =  distances[i + (j - 1)*(length+1)];
                int dist_b = distances[(i - 1) + j*(length+1)];
                int dist_c = distances[(i - 1) + (j - 1)*(length+1)];

                if (dist_a <= dist_b && dist_a <= dist_c) {
                    distances[i + j*(length+1)] = dist_a + 1;
                }
                else if (dist_b <= dist_a && dist_b <= dist_c) {
                    distances[i + j*(length+1)] = dist_b + 1;
                }
                else {
                    distances[i + j*(length+1)] = dist_c + 1;
                }
            }
        }
    }
  
   int result = distances[length + length*(length+1)];
   free(distances);
   return result;
}
