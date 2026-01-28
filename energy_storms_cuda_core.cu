#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "energy_storms.h"

#define BLOCKSIZE 32 // the number of threads per block 

// Wrapper to CUDA calls, to show errors and avoid having to assign each cuda function to a cudaError_t variable. 
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Improvements for comparison:
// - AoS -> SoA (+++)
// - Using __restrict__ (++)
// - Find max particles per storm Pre Main Loop (+) 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// UTILITY FUNCTIONS ///////////////////////////////////////////////////////////


/**
 * Original update function. 
 * Function to update a single position of the layer 
 */
__device__
static void update( 
    float *layer, 
    int layer_sz, 
    int k, 
    int pos, 
    float energy ) 
{
    int distance = abs(pos - k) + 1;
    float attenuacion = sqrtf((float)distance);
    float energy_k = energy / layer_sz / attenuacion;
    if ( energy_k >= THRESHOLD / layer_sz || energy_k <= -THRESHOLD / layer_sz )
        layer[k] += energy_k;
}


/**
 * Scans all waves to find the one with the most particles.
 * This allows us to allocate device memory once instead of every iteration.
 */
int get_max_particles_count(
    int num_storms, 
    Storm *storms) 
{
    int max_p = 0;
    for (int i = 0; i < num_storms; i++) {
        if (storms[i].size > max_p)
            max_p = storms[i].size;
    }
    return max_p;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// NO SHARED MEMORY IMPLEMENTATIONS /////////////////////////////////////////////

/**
 * Simple bombardment phase management (no shared memory & no SoA) -> Slowest implementation of the bombardment phase.
 * Each thread will update the global memory and the input particles are in AoS format. 
 * Each thread will be responsible for a layer cell and iterate on all the particles of the current wave.
 */
__global__ 
__launch_bounds__(BLOCKSIZE)
void bombardment_kernel(
    float *d_layer, 
    int layer_sz, 
    int *d_particles, 
    int particles_sz)
{
    int layer_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (layer_idx >= layer_sz) return; 

    // Update the layer cell for all particles in the storm
    for (int i = 0; i < particles_sz; i++){
        int position = d_particles[i*2]; // impact position
        float energy = (float)d_particles[i*2 + 1] * 1000.0f; // energy in thousandths of Joule
        update(d_layer, layer_sz, layer_idx, position, energy);
    }
}

/**
 * Like @bombardment_kernel but using SoA.
 * This is actually slower than the other one  
 */
__global__ 
__launch_bounds__(BLOCKSIZE)
void bombardment_kernel_soa(
    float * __restrict__ d_layer, 
    int layer_sz, 
    const int * __restrict__ d_particle_pos, 
    const int * __restrict__ d_particle_val, 
    int particles_sz)
{
    int layer_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (layer_idx >= layer_sz) return;

    for (int i = 0; i < particles_sz; i++) {
        int position = d_particle_pos[i]; 
        float energy = (float)d_particle_val[i] * 1000.0f; 
        update(d_layer, layer_sz, layer_idx, position, energy);
    }
}

/**
 * Simple relaxation phase management (no shared memory).
 * Each thread will be responsible for a layer cell and calculate the average between itself & its 2 neighbours.
 * The 1st and last cells will just copy-paste their own value since they haven't both left & right neighbours. 
 */
__global__ 
__launch_bounds__(BLOCKSIZE)
void relaxation_kernel(
    const float *d_layer_in, 
    float *d_layer_out, 
    int layer_sz)
{
    int layer_idx= threadIdx.x + blockIdx.x * blockDim.x;
    if (layer_idx >= layer_sz) return;

    if (layer_idx <= 0 || layer_idx == layer_sz-1) {
        d_layer_out[layer_idx] = d_layer_in[layer_idx]; // first and last should be saved as they are 
        return;
    }

    d_layer_out[layer_idx] = (d_layer_in[layer_idx-1] + d_layer_in[layer_idx] +  d_layer_in[layer_idx+1]) / 3;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// SHARED MEMORY IMPLEMENTATIONS /////////////////////////////////////////////////////


/**
 * A more advanced bombardment phase management (using shared memory). 
 * Each thread will update the global memory and the input particles are in AoS format. 
 */
__global__ 
__launch_bounds__(BLOCKSIZE)
void bombardment_kernelsh(
    float* __restrict__ d_layer, 
    int layer_sz, 
    int* __restrict__ d_particles, 
    int particles_sz)
{
    extern __shared__ int sh_particles[];

    int layer_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (layer_idx >= layer_sz) return;

    for (int tile = 0; tile < particles_sz; tile += blockDim.x) {

        int tid = threadIdx.x;
        int p = tile + tid;

        if (p < particles_sz) {
            sh_particles[2 * tid]     = d_particles[2 * p];
            sh_particles[2 * tid + 1] = d_particles[2 * p + 1];
        }

        __syncthreads(); // avoid reads on shared mem when it's not fully filled 

        int tile_size = min(blockDim.x, particles_sz - tile);

        for (int j = 0; j < tile_size; j++) {
            int position = sh_particles[2 * j];
            float energy = (float)sh_particles[2 * j + 1] * 1000.0f;
            update(d_layer, layer_sz, layer_idx, position, energy);
        }

        __syncthreads(); 
    }
}


/**
 * A more advanced bombardment phase management (using shared memory & SoA).
 */
__global__ 
__launch_bounds__(BLOCKSIZE)
void bombardment_kernelsh_soa(
    float* __restrict__ d_layer, 
    int layer_sz, 
    const int * __restrict__ d_particle_pos, 
    const int * __restrict__ d_particle_val, 
    int particles_sz)
{
    __shared__ int sh_particle_pos[BLOCKSIZE];
    __shared__ int sh_particle_val[BLOCKSIZE];

    int layer_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (layer_idx >= layer_sz) return;

    for (int tile = 0; tile < particles_sz; tile += blockDim.x) {

        int tidx = threadIdx.x;
        int p = tile + tidx;

        if (p < particles_sz) {
            sh_particle_pos[tidx] = d_particle_pos[p];
            sh_particle_val[tidx] = d_particle_val[p];
        }

        __syncthreads(); // avoid reads on shared mem when it's not fully filled 

        int tile_size = min(blockDim.x, particles_sz - tile);

        for (int j = 0; j < tile_size; j++) {
            int position = sh_particle_pos[j];
            float energy = (float)sh_particle_val[j] * 1000.0f;
            update(d_layer, layer_sz, layer_idx, position, energy);
        }

        __syncthreads(); 
    }
}



/**
 * A more advanced relaxation phase management (shared memory).
 * Each thread will first load into shared memory its own value and if they are at the block's edge, load the ghost values. 
 * Then after synchronizing with the other threads in the block, each thread will update into global memory  
 * the average of itself and neighbours. 
 * The 1st and last cells will just copy-paste their own value since they haven't both left & right neighbours. 
 * 
 * Improvements over @relaxation_kernel:
 * - using shared memory prevents reading from global memory 
 */
__global__ 
__launch_bounds__(BLOCKSIZE)
void relaxation_kernelsh(
    const float* __restrict__ d_layer_in, 
    float* __restrict__ d_layer_out, 
    int layer_sz) 
{
    __shared__ float sh[BLOCKSIZE + 2];

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x + 1;  // +1 because sh[0] is left ghost

    // own thread value loading 
    if (global_idx < layer_sz)
        sh[local_idx] = d_layer_in[global_idx];

    // left ghost loading
    if (threadIdx.x == 0 && global_idx >= 1)
        sh[0] = d_layer_in[global_idx - 1];

    // right ghost loading
    if (threadIdx.x == blockDim.x - 1 && global_idx <= layer_sz - 2)
        sh[local_idx + 1] = d_layer_in[global_idx + 1];

    __syncthreads(); // ensures shared memory is filled before accessing it. 

    if (global_idx >= layer_sz) return; // out of bounds

    // just copy first & last
    if (global_idx == 0 || global_idx == layer_sz - 1){
        d_layer_out[global_idx] = d_layer_in[global_idx];
        return;
    }

    d_layer_out[global_idx] = (sh[local_idx - 1] + sh[local_idx] + sh[local_idx + 1]) / 3.0f;
}

__global__ 
__launch_bounds__(BLOCKSIZE)
void maxval_kernelsh(
    float* __restrict__ d_layer, 
    int layer_sz, 
    float* __restrict__ d_block_vals, 
    int* __restrict__ d_block_idxs) 
{
    extern __shared__ char sh[];
    float* s_vals = (float*)sh;
    int* s_idxs = (int*)(s_vals + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float local_val = -FLT_MAX;
    int local_idx = -1;

    // Consider only local max  
    if (gid > 0 && gid < layer_sz - 1) {
        float val = d_layer[gid];
        float left = d_layer[gid - 1];
        float right = d_layer[gid + 1];

        if (val > left && val > right) {
            local_val = val;
            local_idx = gid;
        }
    }

    s_vals[tid] = local_val;
    s_idxs[tid] = local_idx;
    __syncthreads();

    // Reduction 
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_vals[tid + s] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s];
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    // Write back to memory for this block 
    if (tid == 0) {
        d_block_vals[blockIdx.x] = s_vals[0];
        d_block_idxs[blockIdx.x] = s_idxs[0];
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// CORE /////////////////////////////////////////////////////////////////


void core(
    int layer_size, 
    int num_storms, 
    Storm *storms, 
    float *maximum, 
    int *positions) 
{
    int i, k;
    float *h_layer;

    bool useSoA = true; // optimal: true
    bool useSoaPinnedMemory = false; // optimal: true
    bool useBombardmentSharedMem = false; // optimal: true
    bool useRelaxationSharedMem = false; // optimal: false (no real changes)
    bool useRelaxationSwap = true; // optimal: false (no real changes on gtx970)
    bool useMaxvalSequential = false; // optimal: true (no real changes on gtx970)

    // Sequential code 
    if (useMaxvalSequential){
        h_layer = (float *)malloc(sizeof(float) * layer_size);
        if (h_layer == NULL) {
            fprintf(stderr,"Error: Allocating the layer memory\n");
            exit(EXIT_FAILURE);
        }
    }

    float *d_layer;  float *d_layer_copy;
    CUDA_CHECK(cudaMalloc(&d_layer, sizeof(float) * layer_size)); // one layer for the whole execution
    CUDA_CHECK(cudaMalloc(&d_layer_copy, sizeof(float) * layer_size));
    CUDA_CHECK(cudaMemset(d_layer, 0, sizeof(float) * layer_size));  // avoid the previous looping for initialization

    // Grid, Block & Shared Memory sizes 
    dim3 block(BLOCKSIZE);
    dim3 grid((layer_size + BLOCKSIZE -1) / BLOCKSIZE);
    size_t bomb_shmem = 2 * BLOCKSIZE * sizeof(int);
    size_t maxval_shmem = BLOCKSIZE * sizeof(float) + BLOCKSIZE * sizeof(int); 

    ///////////////////////// Maxval Alloc (start)
    float *d_block_max_vals; int *d_block_max_idxs; 
    float *h_block_vals; int *h_block_idxs;
    if (!useMaxvalSequential){
        CUDA_CHECK(cudaMalloc(&d_block_max_vals, grid.x * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_max_idxs, grid.x * sizeof(int)));
        h_block_vals = (float*)malloc(grid.x * sizeof(float));
        h_block_idxs = (int*)malloc(grid.x * sizeof(int));
    }
    ///////////////////////// Maxval Alloc (end)

    ///////////////////////// Max Particles Optimization (start)
    int max_particles = get_max_particles_count(num_storms, storms);
    int *d_particles;
    CUDA_CHECK(cudaMalloc(&d_particles, sizeof(int) * max_particles * 2));
    ///////////////////////// Max Particles Optimization (end)

    int *d_particle_pos; int *d_particle_val; 
    int *h_temp_pos; int *h_temp_val;
    ///////////////////////// AoS -> SoA Optimization (start)
    if (useSoA){
        CUDA_CHECK(cudaMalloc(&d_particle_pos, sizeof(int) * max_particles));
        CUDA_CHECK(cudaMalloc(&d_particle_val, sizeof(int) * max_particles));


        if (!useSoaPinnedMemory){
            // NON-PINNED MEMORY
            h_temp_pos = (int*)malloc(sizeof(int) * max_particles);
            h_temp_val = (int*)malloc(sizeof(int) * max_particles);
        } else {
            // PINNED MEMORY
            CUDA_CHECK(cudaMallocHost(&h_temp_pos, sizeof(int) * max_particles));
            CUDA_CHECK(cudaMallocHost(&h_temp_val, sizeof(int) * max_particles));
        }
    }
    ///////////////////////// AoS -> SoA Optimization (end)


    ////////////////////////////////////////////////////////////////////// SIMULATION BEGINS 
    for( i=0; i<num_storms; i++) {
        int n_particles = storms[i].size;

        //////////////////////////////////////////////////////////////////
        // Bombardment Phase

        // ==== AoS Version
        if (!useSoA){
            CUDA_CHECK(cudaMemcpy(d_particles, storms[i].posval, sizeof(int) * n_particles * 2, cudaMemcpyHostToDevice));
            if (!useBombardmentSharedMem) bombardment_kernel<<<grid, block>>>(d_layer, layer_size, d_particles, storms[i].size);
            else bombardment_kernelsh<<<grid, block, bomb_shmem>>>(d_layer, layer_size, d_particles, storms[i].size);
        }
        // ====

        // ==== SoA Version 
        if (useSoA){
            for (int p = 0; p < n_particles; p++) {
                h_temp_pos[p] = storms[i].posval[p * 2];     // Position
                h_temp_val[p] = storms[i].posval[p * 2 + 1]; // Value
            }
            CUDA_CHECK(cudaMemcpy(d_particle_pos, h_temp_pos, sizeof(int) * n_particles, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_particle_val, h_temp_val, sizeof(int) * n_particles, cudaMemcpyHostToDevice));
            if (!useBombardmentSharedMem) bombardment_kernel_soa<<<grid, block>>>(d_layer, layer_size, d_particle_pos, d_particle_val, n_particles); // no shared-mem (SoA)
            else bombardment_kernelsh_soa<<<grid, block>>>(d_layer, layer_size, d_particle_pos, d_particle_val, storms[i].size);
        }
        // ====

        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaDeviceSynchronize());  
        //////////////////////////////////////////////////////////////////


        //////////////////////////////////////////////////////////////////
        // Relaxation Phase
        if (!useRelaxationSwap){
            CUDA_CHECK(cudaMemcpy(d_layer_copy, d_layer, sizeof(float) * layer_size, cudaMemcpyDeviceToDevice)); 
            if (!useRelaxationSharedMem) relaxation_kernel<<<grid, block>>>(d_layer_copy, d_layer, layer_size);
            else relaxation_kernelsh<<<grid, block>>>(d_layer_copy, d_layer, layer_size);
        } else {
            if (!useRelaxationSharedMem) relaxation_kernel<<<grid, block>>>(d_layer, d_layer_copy, layer_size);
            else relaxation_kernelsh<<<grid, block>>>(d_layer, d_layer_copy, layer_size);
            // Pointer Swap (no cudaMemcpy before relaxation phase)
            float *tmp = d_layer;
            d_layer = d_layer_copy;
            d_layer_copy = tmp;
        }

        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaDeviceSynchronize());  
        //////////////////////////////////////////////////////////////////


        //////////////////////////////////////////////////////////////////
        // Maximum Value Location Phase 
        if (!useMaxvalSequential) {
            maxval_kernelsh<<<grid, block, maxval_shmem>>>(d_layer, layer_size, d_block_max_vals, d_block_max_idxs);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Partial results
            CUDA_CHECK(cudaMemcpy(h_block_vals, d_block_max_vals, grid.x * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_block_idxs, d_block_max_idxs, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

            // Final reduction on CPU 
            float current_max = -FLT_MAX; 
            int current_pos = -1;

            for(int b = 0; b < grid.x; b++) {
                if (h_block_vals[b] > current_max) {
                    current_max = h_block_vals[b];
                    current_pos = h_block_idxs[b];
                }
            }
        
            if (current_pos != -1) {
                 maximum[i] = current_max;
                 positions[i] = current_pos;
            }
        } else {

        // ==== SEQ Version 
            CUDA_CHECK(cudaMemcpy(h_layer, d_layer, sizeof(float) * layer_size, cudaMemcpyDeviceToHost));   
            for(k=1; k<layer_size-1; k++) {
                if ( h_layer[k] > h_layer[k-1] && h_layer[k] > h_layer[k+1] ) {
                    if ( h_layer[k] > maximum[i] ) {
                        maximum[i] = h_layer[k];
                        positions[i] = k;
                    }
                }
            }
        // ==== 
        }
        //////////////////////////////////////////////////////////////////
    }
    ////////////////////////////////////////////////////////////////////// SIMULATION ENDS


    //////////////////////////////////////////////////////////////////
    // Cleanup
    if (!useSoA){
        CUDA_CHECK(cudaFree(d_particles));
    }

    CUDA_CHECK(cudaFree(d_layer));
    CUDA_CHECK(cudaFree(d_layer_copy));

    if (!useMaxvalSequential){
        CUDA_CHECK(cudaFree(d_block_max_vals));
        CUDA_CHECK(cudaFree(d_block_max_idxs));
        free(h_block_vals);
        free(h_block_idxs);
    } else {
        free(h_layer);
    }

    if (useSoaPinnedMemory && useSoA){
        CUDA_CHECK(cudaFreeHost(h_temp_pos));
        CUDA_CHECK(cudaFreeHost(h_temp_val));
    } else if (useSoA && !useSoaPinnedMemory){
        free(h_temp_pos);
        free(h_temp_val);
    }

}
