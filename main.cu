#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <curand.h> // generate random numbers
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

// reference https://youtu.be/MVutNZaNTkM?si=MqROqOtXDO84596L
// cuBLAS assumes column major order

void
verifysolution(float *a, float *b, float *c, int n)
{
    float temp;
    float epsilon = 0.001;

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            temp=0;
            for(int k=0; k < n; k++){
                temp += a[k*n + i] * b[j*n+k];
            }
            assert(fabs(c[j*n+i] - temp) < epsilon);
        }
    }
}

int 
main()
{
    clock_t start = clock();
    int n = 1 << 11; // bits shift by 10 places
    size_t bytes = n * n * sizeof(float);

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

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // scaling factors

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    clock_t end = clock();
    double cuda_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CUDA Execution time: %f seconds\n", cuda_time_used);

    clock_t cpu_start = clock();
    verifysolution(h_a, h_b, h_c, n);
    clock_t cpu_end = clock();
    double cpu_time_used = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU Execution time: %f seconds\n", cpu_time_used);
    printf("solution is correct\n");

    return 0;






}