#pragma once
#include <omp.h>
#include <stdio.h>
#include <math.h>

#define ARR_SIZE 1000000
#define NUMS_IN_RANGE 256

void test(int *data, int n);
void test2(int *arr, int *histogram);
void randArr(int *arr, int numElements);
int *computeHistogramOnGPU(int *data, int numElements);
