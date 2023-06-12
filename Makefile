build:
	mpicxx -fopenmp -Wall -Wextra -c main.c -o main.o
	mpicxx -fopenmp -Wall -Wextra -c cFunctions.c -o cFunctions.o
	nvcc --expt-relaxed-constexpr -Xcompiler -Wall -Xcompiler -Wextra -I./Common  -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -Wall -Wextra -o mpiCudaOpemMP  main.o cFunctions.o cudaFunctions.o  -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
	

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP
