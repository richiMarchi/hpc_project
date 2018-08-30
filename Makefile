## Questo e' un frammento del Makefile utilizzato negli esempi
## illustrati durante il corso. Se necessario questo file puo' essere
## modificato e personalizzato in base alle proprie esigenze.
##
## Questo Makefile compila automaticamente tutti i file "omp-*.c"
## assumendo che si tratti di programmi OpenMP (usando il flag
## -fopenmp); i file "cuda-*.cu" vengono compilati con il compilatore
## nvcc; infine, i file "mpi-*.c" vengono compilati con mpicc.

EXE_OMP:=$(basename $(wildcard omp-*.c))
EXE_MPI:=$(basename $(wildcard mpi-*.c))
EXE_CUDA:=$(basename $(wildcard cuda-*.cu))
EXE_SERIAL:=ltl-skel gen-input
EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_CUDA)
CFLAGS+=-std=c99 -Wall -Wpedantic
NVCC?=nvcc
MPICC?=mpicc
NVCFLAGS+=-Wno-deprecated-gpu-targets

ALL: mpi openmp serial cuda

$(EXE_OMP): CFLAGS+=-fopenmp 
$(EXE_OMP): LDLIBS+=-lgomp -lrt
openmp: $(EXE_OMP)

$(EXE_MPI): CC=$(MPICC)
mpi: $(EXE_MPI)

serial: $(EXE_SERIAL)

cuda: $(EXE_CUDA)

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

clean:
	\rm -f $(EXE) *.o *~
