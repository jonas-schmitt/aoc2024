#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel for one pass of bubble sort
__global__ void bubble_sort_pass(int* arr, size_t n, int pass) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread handles one comparison
    // Only compare even or odd indices based on pass number to avoid race conditions
    if (idx < n - 1 && (idx % 2) == (pass % 2)) {
        if (arr[idx] > arr[idx + 1]) {
            int temp = arr[idx];
            arr[idx] = arr[idx + 1];
            arr[idx + 1] = temp;
        }
    }
}

__global__ void pairwise_dist(int *a, int* b, int *res, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        res[i] = abs(a[i] - b[i]);
    }
}

// Parallel reduction kernel using shared memory
__global__ void reduce_array(int *input, int *output, size_t n) {
    extern __shared__ int sdata[];
    
    // Load shared mem from global mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = 0;
    if (i < n) {
        sdata[tid] = input[i];
    }
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
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
    int *d_left, *d_right, *d_dist, *d_temp, *d_res;
    
    cudaMalloc(&d_left, n * sizeof(int));
    cudaMalloc(&d_right, n * sizeof(int));
    cudaMalloc(&d_dist, n * sizeof(int));
    cudaMalloc(&d_temp, ((n + 255) / 256) * sizeof(int));  // For intermediate results
    cudaMalloc(&d_res, sizeof(int));
    
    cudaMemcpy(d_left, left.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Sort arrays
    for (int pass = 0; pass < n; pass++) {
        bubble_sort_pass<<<blocksPerGrid, threadsPerBlock>>>(d_left, n, pass);
        bubble_sort_pass<<<blocksPerGrid, threadsPerBlock>>>(d_right, n, pass);
        cudaDeviceSynchronize();
    }
    
    // Calculate pairwise distances
    pairwise_dist<<<blocksPerGrid, threadsPerBlock>>>(d_left, d_right, d_dist, n);
    
    // First reduction pass
    reduce_array<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(
        d_dist, d_temp, n);
    
    // Second reduction pass if needed
    if (blocksPerGrid > 1) {
        reduce_array<<<1, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(
            d_temp, d_res, blocksPerGrid);
    } else {
        cudaMemcpy(d_res, d_temp, sizeof(int), cudaMemcpyDeviceToDevice);
    }
    
    int res;
    cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << res << std::endl;
    
    // Free device memory
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_dist);
    cudaFree(d_temp);
    cudaFree(d_res);
    
    return 0;
}