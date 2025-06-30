/*
 * @Author: yykzjh
 * @Date: 2025-06-24 17:34:36
 * @LastEditors: yykzjh
 * @LastEditTime: 2025-06-24 18:48:13
 * @FilePath: /CUDA-CPlusPlus-Programming-Guide/chapter_05/vector_add.cu
 * @Description: 本示例是一个非常基本的示例，实现了逐元素的向量加法。
 * 
 * Copyright (c) 2025 by yykzjh, All Rights Reserved. 
 */
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
// For helper functions (error checking, etc.)
#include <helper_cuda.hpp>


#define CHECK_CUDA(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "Failed to execute cuda runtime API (error code %s) at %s:%d!\n", \
                    cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


const int N = 20;

/**
 * @description: CUDA kernel for vector add
 * @return {*}
 */
__global__ void vector_add(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;

    C[i][j] = A[i][j] + B[i][j] + 1.0f;
}

int main(void)
{

    // 初始化要计算的数组长度
    size_t num_bytes = N * N * sizeof(float);
    printf("Vector addition of %d elements.\n", N * N);

    // 分配输入数组A和B的内存
    float* h_A = (float*)malloc(num_bytes);
    float* h_B = (float*)malloc(num_bytes);
    // 分配输出数组C的内存
    float* h_C = (float*)malloc(num_bytes);

    // 验证分配内存成功
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // 初始化vectors的数据
    for (int i=0;i<N*N;++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // 分配vevtor A、B和C在device上的内存
    float* d_A = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, num_bytes));
    float* d_B = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_B, num_bytes));
    float* d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_C, num_bytes));

    // 将输入vectors中的数据从host复制到device上
    printf("Copy input data from the host memory to the CUDA device\n");
    CHECK_CUDA(cudaMemcpy(d_A, h_A, num_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, num_bytes, cudaMemcpyHostToDevice));

    // 启动内核函数
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    vector_add<<<numBlocks, threadsPerBlock>>>((float (*)[N])d_A, (float (*)[N])d_B, (float (*)[N])d_C);
    auto err = cudaGetLastError();
    CHECK_CUDA(err);

    // 从device内存上复制计算结果到host的内存
    printf("Copy output data from the CUDA device to the host memory\n");
    CHECK_CUDA(cudaMemcpy(h_C, d_C, num_bytes, cudaMemcpyDeviceToHost));

    // 验证结果的正确性
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_A[i] + h_B[i] + 1 - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    // 释放device上分配的内存
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    // 释放host上分配的内存
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}
