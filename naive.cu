#include <cuda_runtime.h>
#include <curand.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// naive code to do matrix multiplication 1024*1024

__global__
void
sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < M && y < M) {
        float tmp = 0.0;
        for(int i=0; i<K; i++){
            tmp += A[x * K + i] * B[i * N + y];
        }

        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int
main()
{
    int n = 1 << 10;

    size_t bytes = n * n *  sizeof(float);
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT); // default random number generator

    // generate seed

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    //filling with random numbers

    curandGenerateUniform(prng, d_a, n*n);
    curandGenerateUniform(prng, d_b, n*n);

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    dim3 gridDim(32,32,1);
    dim3 blockDim(32,32,1);
    
    clock_t start = clock();
    sgemm_naive<<<gridDim,blockDim>>>(M,N,K,1.0f,d_a,d_b,0.0f,d_c);
    
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    clock_t end = clock();

    free(h_a);
    free(h_b);
    free(h_c);
    free(d_a);
    free(d_b);
    free(d_c);

    double cuda_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("naive Execution time: %f seconds\n", cuda_time_used);
} 