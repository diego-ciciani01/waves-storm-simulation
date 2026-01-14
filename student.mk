# --- Execution Parameters ---
export OMP_NUM_THREADS=4
MPI_PROCS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# --- MPI Run Flags ---
MPIRUN_FLAGS = -np $(MPI_PROCS) \
               --bind-to none

# --- Compiler Flags ---
# Flags for MPI+OpenMP code
# Uncomment and add extra flags if you need them
#MPI_OMP_EXTRA_CFLAGS =
MPI_OMP_EXTRA_LIBS = -march=native \
					 -ffast-math

# Flags for CUDA code
# Uncomment and add extra flags if you need them
#CUDA_EXTRA_CFLAGS =
#CUDA_EXTRA_LIBS =
