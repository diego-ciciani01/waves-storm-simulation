# --- Execution Parameters ---
export OMP_NUM_THREADS=4
MPI_PROCS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export ONE_BIG_FILE=
# --- MPI Run Flags ---
MPIRUN_FLAGS = -np $(MPI_PROCS) \
               --bind-to core
# --- Compiler Flags ---
# Flags for MPI+OpenMP code
# Uncomment and add extra flags if you need them
#MPI_OMP_EXTRA_CFLAGS =
MPI_OMP_EXTRA_LIBS = -march=native \
					 -ffast-math

# Flags for CUDA code
# Uncomment and add extra flags if you need them
# CUDA_EXTRA_CFLAGS = -gencode arch=compute_52,code=sm_52 # allow run on gtx 970 
CUDA_EXTRA_CFLAGS = -O3 
#CUDA_EXTRA_LIBS =
