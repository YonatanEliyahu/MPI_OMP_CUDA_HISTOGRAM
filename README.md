# MPI_OMP_CUDA_HISTOGRAM

A histogram is a graphical representation that shows the frequency or count of data within specific intervals or bins, providing a visual summary of the data distribution.

Simple MPI+OpenMP+CUDA Integration.
The MASTER process initially array sized ARR_SIZE and set it with random values.
It sends the half of the array to the process 1 (SLAVE).
Both will calculate the HISTOGRAM of thier part using CUDA.
The MASTER process will sum the histograms and will run two tests to check for correctness. 

