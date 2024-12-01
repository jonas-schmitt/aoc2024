#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel for counting matching elements
__global__ void count_elements(int *a, int *b, int *counts, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        counts[idx] = 0;
        for(size_t i = 0; i < n; ++i) {
            if(a[idx] == b[i]) {
                ++counts[idx];
            }
        }
    }
}

// CUDA kernel to compute sum with block-level reduction
__global__ void compute_sum(int *a, int *counts, int *result, size_t n) {
    extern __shared__ int sdata[];  // dynamically allocated shared memory
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and compute product
    int local_sum = 0;
    if (idx < n) {
        local_sum = a[idx] * counts[idx];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

int main(int argc, char **argv) {
    int a, b;
    std::vector<int> left, right;
    
    // Read input values
    while (std::cin >> a >> b) {
        left.push_back(a);
        right.push_back(b);
    }
    
    size_t n = left.size();
    int *d_left, *d_right, *d_counts, *d_result;
    cudaMalloc(&d_left, n * sizeof(int));
    cudaMalloc(&d_right, n * sizeof(int));
    cudaMalloc(&d_counts, n * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    cudaMemcpy(d_left, left.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Initialize result to 0
    int zero = 0;
    cudaMemcpy(d_result, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    // Query device for optimal block size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const size_t threadsPerBlock = std::min(static_cast<size_t>(prop.maxThreadsPerBlock), 
                                          static_cast<size_t>(1024));  // typical max is 1024
    const size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Count matching elements
    count_elements<<<blocksPerGrid, threadsPerBlock>>>(d_left, d_right, d_counts, n);
    
    // Compute sum with block-level reduction
    compute_sum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(
        d_left, d_counts, d_result, n);
    
    // Copy result back to host
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print result
    std::cout << result << std::endl;
    
    // Free device memory
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_counts);
    cudaFree(d_result);
    
    return 0;
}