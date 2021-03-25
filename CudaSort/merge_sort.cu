#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t mergeSortWithCuda(int* arr, unsigned int size);

__global__ void mergeSortKernel(int *arr, int *aux, unsigned int blockSize, const unsigned int last)
{
    int x = threadIdx.x;
    int start = blockSize * x;
    int end = start + blockSize - 1;
    int mid = start + (blockSize / 2) - 1;
    int l = start, r = mid + 1, i = start;

    if (end > last) { end = last; }
    if (start == end || end <= mid) { return; }

    while (l <= mid && r <= end) {
        if (arr[l] <= arr[r]) {
            aux[i++] = arr[l++];
        }
        else {
            aux[i++] = arr[r++];
        }
    }

    while (l <= mid) { aux[i++] = arr[l++]; }
    while (r <= end) { aux[i++] = arr[r++]; }

    for (i = start; i <= end; i++) {
        arr[i] = aux[i];
    }
}

cudaError_t mergeSortWithCuda(int *arr, unsigned int size)
{
    int *dev_arr = 0;
    int *dev_aux = 0;
    const unsigned int last = size - 1;
    unsigned int threadCount;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (main and aux array).
    cudaStatus = cudaMalloc((void**)&dev_arr, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_aux, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    for (unsigned int blockSize = 2; blockSize < 2 * size; blockSize *= 2) {
        threadCount = size / blockSize;
        if (size % blockSize > 0) { threadCount++; }

        // Launch a kernel on the GPU with one thread for each block.
        mergeSortKernel<<<1, threadCount>>>(dev_arr, dev_aux, blockSize, last);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "mergeSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mergeSortKernel!\n", cudaStatus);
            goto Error;
        }
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(arr, dev_arr, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_arr);
    cudaFree(dev_aux);

    return cudaStatus;
}

