# High-Energy Particle Storms Simulation
**Multicore Programming Exam 2025/2026**

## Project Description
This project simulates the effects of high-energy particle bombardment on an exposed surface, such as the hull of a space vessel. The simulation considers a cross-section of the surface represented by a discrete number of control points.



The simulation follows a three-stage process for each wave of particles:
1.  **Bombardment**: Particles impact the surface, transferring energy to the impact point and its neighborhood with distance-based attenuation.
2.  **Relaxation**: The material redistributes its charge by averaging the energy value of each point with its immediate neighbors.
3.  **Analysis**: The program identifies and reports the point with the highest accumulated energy for each wave.

---

## 🛠 Compilation and Build System
The project uses a `Makefile` to manage different parallel implementations. Use the following commands to build the targets:

### General Commands
* `make help`: Shows the help menu with all available options.
* `make all`: Builds all versions (Sequential, MPI+OpenMP, and CUDA).
* `make clean`: Removes all compiled binaries and temporary files.
* `make debug`: Builds all versions with the `-DDEBUG` flag, enabling graphical output for small arrays (size $\le 35$).

### Specific Targets
* `make energy_storms_seq`: Builds the **Sequential** baseline version.
* `make energy_storms_mpi_omp`: Builds the hybrid **MPI + OpenMP** version.
* `make energy_storms_cuda`: Builds the **CUDA** (GPU) version.

---

## Implementations

### 1. Sequential (`energy_storms_seq`)
The baseline implementation where all phases (bombardment, relaxation, and maximum search) are executed on a single core.

### 2. Hybrid MPI + OpenMP (`energy_storms_mpi_omp`)
Designed for distributed computing:
* **MPI**: Used for domain decomposition, partitioning the control point array across different nodes.
* **OpenMP**: Used within each MPI process to parallelize the bombardment and relaxation loops across multiple CPU cores.
* **Synchronization**: Ensures data consistency during the relaxation phase, where neighboring values are required for averaging.

### 3. CUDA (`energy_storms_cuda`)
Leverages GPU acceleration:
* **Parallel Kernels**: High-throughput computation of energy attenuation for thousands of particles.
* **Memory Management**: Efficient data transfer between Host and Device and use of fast on-chip memory for relaxation stencils.

---

## 📖 Usage
To run the simulation, use the following syntax:

```bash
./energy_storms_<version> <size> <wave_file1> [wave_file2 ...]
