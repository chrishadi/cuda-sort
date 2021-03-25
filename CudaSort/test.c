#include "merge_sort.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool assertArrEq(int* expected, int* actual, size_t size);

int cmpInt(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

int main()
{
    const unsigned int count = rand() % 2048;
    const unsigned int size = count * sizeof(int);
    int* arr = (int*) malloc(size);
    int* expected = (int*) malloc(size);

    for (int i = 0; i < count; i++) {
        expected[i] = arr[i] = rand();
    }

    qsort(expected, count, sizeof(int), cmpInt);

    cudaError_t cudaStatus = mergeSortWithCuda(arr, count);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mergeSortWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    if (!assertArrEq(expected, arr, size)) {
        return 1;
    }

    printf("test ok.");

    free(expected);
    free(arr);

    return 0;
}

bool assertArrEq(int* expected, int* actual, size_t size) {
    if (memcmp(expected, actual, size) != 0) {
        printf("actual array is not equal to the expected array!");
        return false;
    }

    return true;
}

