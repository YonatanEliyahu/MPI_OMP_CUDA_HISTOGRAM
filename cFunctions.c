#include "myProto.h"

void randArr(int *arr, int numElements)
{
    // the following function will fill the arr in random values in range of [0 - (NUMS_IN_RANGE-1)]
#pragma omp parallel for
    for (int index = 0; index < numElements; index++)
    {
        arr[index] = rand() % NUMS_IN_RANGE;
    }
}

void test(int *histogram, int n)
{
    // the following function will sum the histogram arr and check if the sum of the histogram is equals to ARR_SIZE
    int i, sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (i = 0; i < n; i++)
    {
        sum += histogram[i];
    }
    if (sum == ARR_SIZE)
        printf("The test passed successfully\n");
    else
        printf("The test didn't passed successfully\n");
}

void test2(int *arr, int *histogram)
{
    // the following function will calculate the sum of the arr by the treditional way and by histogram arr and check if the sums are equals
    int i, sum1 = 0, sum2 = 0;
#pragma omp parallel for reduction(+ : sum1)
    for (i = 0; i < NUMS_IN_RANGE; i++)
    {
        sum1 += histogram[i] * i;
    }
#pragma omp parallel for reduction(+ : sum2)
    for (i = 0; i < ARR_SIZE; i++)
    {
        sum2 += arr[i];
    }
    if (sum1 == sum2)
        printf("The test passed successfully\n");
    else
        printf("The test didn't passed successfully\n");
}
