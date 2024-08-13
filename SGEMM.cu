// each SM (streaming multiprocessor) has a shared memory.
// threads in a block can communicate with each other.
// approach - load chunk in the shared memory and perform complete operation on them 

#include <cuda_runtime.h>
#include <curand.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

__global__
void
smem_cache(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    unsigned int cRow = blockIdx.x;
    unsigned int cCol = blockIdx.y;

    __shared__ float As[32*32];
    __shared__ float Bs[32*32]; // shared memory blocks

    unsigned int threadCol = threadIdx.x % 32;
    unsigned int threadRow = threadIdx.x / 32; 

    A += cRow * 32 * K; // going to the correct rows using pointer addition
    B += cCol * 32;

    float tmp = 0.0;
    for(int i=0; i<K; i += 32) {
        As[threadRow * 32 + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * 32 + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();
        A += 32;
        A += 32;

        for (int dotIdx = 0; dotIdx < 32; ++dotIdx) {
            tmp += As[threadRow * 32 + dotIdx] *
            Bs[dotIdx * 32 + threadCol];
        }

        __syncthreads();

        C[threadRow * N + threadCol] =
        alpha * tmp + beta * C[threadRow * N + threadCol];
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
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT); 


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
    smem_cache<<<gridDim,blockDim>>>(M,N,K,1.0f,d_a,d_b,0.0f,d_c);
    
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
    printf("SGEMM Execution time: %f seconds\n", cuda_time_used);
}