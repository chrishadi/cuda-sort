#include "merge_sort.cuh"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OK 1
#define EXPECTATION_ERROR 1
#define MALLOC_ERROR 2
#define CUDA_ERROR 3

bool assertArrEq(int* expected, int* actual, size_t size);
int testMergeSortWithCuda(int* actual, int* expected, const unsigned int count);

int main()
{
    const unsigned int count = rand() % 2048;
    const unsigned int size = count * sizeof(int);
    int status = MALLOC_ERROR;
    int* actual = (int*) malloc(size);
    int* expected = (int*) malloc(size);

    if (actual != NULL && expected != NULL) {
        status = testMergeSortWithCuda(actual, expected, count);
    }
    else {
        fprintf(stderr, "malloc failed!");
    }

    free(actual);
    free(expected);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    int cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return CUDA_ERROR;
    }

    return status;
}

int cmpInt(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

int testMergeSortWithCuda(int* actual, int* expected, const unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        expected[i] = actual[i] = rand();
    }

    qsort(expected, count, sizeof(int), cmpInt);

    cudaError_t cudaStatus = mergeSortWithCuda(actual, count);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mergeSortWithCuda failed!");
        return CUDA_ERROR;
    }

    if (!assertArrEq(expected, actual, count * sizeof(int))) {
        puts("cuda sorted array is not equal to the qsorted array!");
        return EXPECTATION_ERROR;
    }

    puts("test ok.");
    return OK;
}

bool assertArrEq(int* expected, int* actual, size_t size) {
    if (memcmp(expected, actual, size) != 0) {
        return false;
    }

    return true;
}
