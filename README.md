# MPI_OMP_CUDA_HISTOGRAM

Simple MPI+OpenMP+CUDA Integration.
The MASTER process initially array sized ARR_SIZE and set it with random values.
It sends the half of the array to the process 1 (SLAVE).
Both will calculate the HISTOGRAM of thier part using CUDA.
The MASTER process will sum the histograms and will run two tests to check for correctness. 

