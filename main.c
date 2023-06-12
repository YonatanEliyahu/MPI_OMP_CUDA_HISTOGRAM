#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "myProto.h"

#define MASTER 0
#define SLAVE 1

/*
Simple MPI+OpenMP+CUDA Integration
The MASTER process initially array sized ARR_SIZE and set it with random values.
It sends the half of the array to the process 1 (SLAVE).
Both will calculate the HISTOGRAM of thier part using CUDA.
The MASTER process will sum the histograms and will run two tests to check for correctness. 
*/

int main(int argc, char *argv[])
{
   int size, rank, i;
   int *data = NULL;
   MPI_Status status;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   if (size != 2)
   {
      printf("Run the program with two processes only\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int chunkSize = ARR_SIZE / size;

   if (rank == MASTER) // master work - initialize the arr and set values in the arr
   {
      srand(time(NULL));
      // Allocate memory for the whole array and send a half of the array to other process
      if ((data = (int *)malloc(ARR_SIZE * sizeof(int))) == NULL)
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      randArr(data, ARR_SIZE);
   }
   // global work - MASTER and SLAVE
   //  Allocate memory and reieve a half of array for the scatter
   int *chunk;
   if ((chunk = (int *)malloc((chunkSize) * sizeof(int))) == NULL)
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   MPI_Scatter(data, chunkSize, MPI_INT, chunk, chunkSize, MPI_INT, MASTER, MPI_COMM_WORLD);

   int *privateHistogram;
   // On each process - perform a second half of its task with CUDA
   privateHistogram = computeHistogramOnGPU(chunk, chunkSize);
   if (privateHistogram == NULL)
      MPI_Abort(MPI_COMM_WORLD, __LINE__);

   if (rank == SLAVE) // sending result of the second side histogram to the master
   {
      MPI_Send(privateHistogram, NUMS_IN_RANGE, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
   }
   else // MASTER
   {
      // getting the secong half histogram from the SLAVE
      int *finalHistogram = (int *)malloc(sizeof(int *) * NUMS_IN_RANGE);
      if (finalHistogram == NULL)
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      MPI_Recv(finalHistogram, NUMS_IN_RANGE, MPI_INT, SLAVE, 0, MPI_COMM_WORLD, &status);
      // sum the histograms
#pragma omp parallel for
      for (i = 0; i < NUMS_IN_RANGE; i++)
         finalHistogram[i] += privateHistogram[i];
      test(finalHistogram, NUMS_IN_RANGE);
      test2(data,finalHistogram);
   }

   MPI_Finalize();

   return 0;
}
