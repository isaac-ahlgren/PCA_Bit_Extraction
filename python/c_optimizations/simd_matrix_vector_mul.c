#include "headers.h"
#include <immintrin.h>

inline __m256 mul8( const float* p1, const float* p2, unsigned int offsetRegs )
{
    int lanes = offsetRegs * 8;
    const __m256 a = _mm256_loadu_ps( p1 + lanes );
    const __m256 b = _mm256_loadu_ps( p2 + lanes );
    return _mm256_mul_ps( a, b );
}

inline __m256 fma8( __m256 acc, const float* p1, const float* p2, unsigned int offsetRegs )
{
    unsigned int lanes = offsetRegs * 8;
    const __m256 a = _mm256_loadu_ps( p1 + lanes );
    const __m256 b = _mm256_loadu_ps( p2 + lanes );
    return _mm256_fmadd_ps( a, b, acc );
}

// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
inline float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

void matrix_vec_mult_avx(float *mat, uint32_t dim_size, float *vec, float *new_vec) {
    float *row = mat, *matend = mat + dim_size*dim_size;
    __m256 dot0;

    uint32_t non_simd_dim_size = dim_size % 8;
    uint32_t simd_dim_size = dim_size - non_simd_dim_size;
    int k = 0;
    while(row < matend) {
        float *col = row;
        float *vecptr = vec;

        dot0 = mul8(col,vecptr,0);
        vecptr += 8;
        col += 8;
        // do portion of matrix multiplication possible with simd
        while(vecptr < vec + simd_dim_size) {
            dot0 = fma8(dot0,col,vecptr,0);
            vecptr += 8;
            col += 8;
        }
        new_vec[k] = sum8(dot0);

        // do portion of matrix multiplication not possible with simd
        while(vecptr < vec + simd_dim_size + non_simd_dim_size) {
            new_vec[k] += *col * *vecptr;
            vecptr++;
            col++;
        }
        k++;
        row += dim_size; // go to next row of matrix
    }
    
}

/*
#define N 9
extern float mat[], vec[];
float test_matrix[N*N];
float test_vector[N];
float new_vec[N];
float new_vec2[N];
int main()
{
	//initalization
	time_t t;
	srand((unsigned) time(&t));
	int i, j;
 	clock_t start, end, start1, end1;
	double cpu_time_used, cpu_time_used1;
	for(i=0; i<N; i++) {
		test_vector[i] = rand();
		//test_vector[i] = 1.;
		for(j=0; j<N; j++){
        		//test_matrix[i+j] = rand();
            		if(i == j) { test_matrix[i*N+j] = 1.; }
		       	else { test_matrix[i*N+j] = 0.; }
		}
	}

	//printing
 	printf("Test Vector:\n");
	for (int q = 0; q < N; q++) { 
		printf("%d, %f\n", q, test_vector[q]); 
    	}
    	printf("Test Matrix:\n");
    	for (int r = 0; r < N; r++) {
		for (int s = 0; s < N; s++) {
			printf("%f ", test_matrix[r*N + s]);
		}
    		printf("\n");
    	}
    	printf("\n");
    	printf("done making test matrix!");
    	
	//x86 or arm gemm
	start = clock();
	matrix_vec_mult_avx(test_matrix, N, test_vector, new_vec);
	end = clock();
        //cpu_time_used = ((double) (end - start))/CLOCKS_PER_SEC*1000;
        //printf("\n%fms x86\n", cpu_time_used);
    	
	//unvectorized gemm
	//start1 = clock();
	//matrix_mult_basic(N, N, test_vector, test_matrix, new_vec2);
	//end1 = clock();
	//cpu_time_used1 =((double) (end1 - start1))/CLOCKS_PER_SEC*1000;
	//printf("\n%fms unvectorized\n", cpu_time_used1);
        
        for (int i = 0; i < N; i++)
        {
           printf("%f ", new_vec[i]);
        }
        printf("\n");

	return 0;
}
*/
