cmake_minimum_required(VERSION 3.18)
project(CudaProject CUDA CXX)

find_package(CUDAToolkit REQUIRED)

add_executable(CudaProject main.cu)

target_link_libraries(CudaProject PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)