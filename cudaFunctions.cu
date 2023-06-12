#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

#define NUM_OF_THREADS 20
#define NUM_OF_BLOCKS 10

__global__ void calcHistogram(const int *arr, int *histogram, int numElements, int numElementsPerThread)
{
    int tid = (blockDim.x * blockIdx.x + threadIdx.x); // computing thread id
    int from = tid * numElementsPerThread;             // computing start of the scope of work for the current thread
    int to = from + numElementsPerThread;              // computing end of the scope of work for the current thread
    if (to > numElements)                              // if the scope calculation is bigger then the real size of the arr
        to = numElements;                              // end scope in the end of the arr

    for (int index = from; index < to; index++)
    {
        atomicAdd(&histogram[tid * NUMS_IN_RANGE + arr[index]], 1); // increasing the sub-histogram of the current tid
    }
}

__global__ void sumHistogram(const int *histogram, int *collective, int numOfSubArrs)
{
    // the folowing function will get huge histogram arr, and will set a collective histogram arr by suming every index of the sub-histogram
    int tid = threadIdx.x;
    for (int index = 0; index < numOfSubArrs; index++)
        collective[tid] += histogram[tid + index * NUMS_IN_RANGE];
}

int *computeHistogramOnGPU(int *data, int numElements)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t size = numElements * sizeof(int);

    // Allocate memory on GPU to copy the data from the host
    int *d_A;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //  Copy the data from host to the GPU memory.
    err = cudaMemcpy(d_A, data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data array from host to device -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size_t histogramSize = NUMS_IN_RANGE * NUM_OF_BLOCKS * NUM_OF_THREADS * sizeof(int);
    // Allocate memory on GPU for the histogram results and setting it to 0
    int *h_A;
    err = cudaMalloc((void **)&h_A, histogramSize);
    err = cudaMemset(h_A, 0, histogramSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Kernel
    int threadsPerBlock = NUM_OF_THREADS;
    int blocksPerGrid = NUM_OF_BLOCKS;
    int numElementsPerThread = numElements / (threadsPerBlock * blocksPerGrid); // calculating the scope of work for each thread
    if (numElements % (threadsPerBlock * blocksPerGrid) != 0)
        numElementsPerThread++; // dealling with leftovers

    calcHistogram<<<blocksPerGrid, threadsPerBlock>>>(d_A, h_A, numElements, numElementsPerThread);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate memory on GPU for the histogram collective result and setting it to 0
    int realHistogramSize = NUMS_IN_RANGE * sizeof(int);
    int *hcr_A = NULL; // histogram collective result
    err = cudaMalloc((void **)&hcr_A, realHistogramSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemset(hcr_A, 0, realHistogramSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Kernel
    threadsPerBlock = NUMS_IN_RANGE;
    blocksPerGrid = 1;

    sumHistogram<<<blocksPerGrid, threadsPerBlock>>>(h_A, hcr_A, NUM_OF_THREADS * NUM_OF_BLOCKS);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the data from GPU to the host memory.
    int *collective_histogram = (int *)malloc(realHistogramSize);
    err = cudaMemcpy(collective_histogram, hcr_A, realHistogramSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free allocated memory on GPU
    if (cudaFree(d_A) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free allocated memory on GPU
    if (cudaFree(h_A) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free allocated memory on GPU
    if (cudaFree(hcr_A) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return collective_histogram;
}
