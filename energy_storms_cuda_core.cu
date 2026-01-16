#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"

#define BLOCKSIZE 256

// Wrapper to CUDA calls, to show errors and avoid having to assign to a cudaError_t variable. 
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// TODO (to test overheads)
// - precomputed lookup table for the sqrt 
// - documentation of kernels
// - remove the h_layer once all phases are parallelized 
// - register accumulation for the bombardment phase, instead of N updates, update once global memory and accumulate N times on the thread. 
// - loop the storms before the main loop, find the max particles and allocate once outside the loop with the MAX particles (avoid multiple allocations) 
// - instead of doing -> CUDA_CHECK(cudaMemcpy(d_layer_copy, d_layer, sizeof(float) * layer_size, cudaMemcpyDeviceToDevice)); relaxation_kernel<<<grid, block>>>(d_layer_copy, d_layer, layer_size);
//                    ...do a pointer swap in each storm iteration.

/**
 * Kernels with sh suffixed to the function name will make use of shared memory and try to optimize  
 * performance. 
 * 
 */


/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
__device__
static void update( float *layer, int layer_size, int k, int pos, float energy ) {
    /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
    int distance = pos - k;
    if ( distance < 0 ) distance = - distance;

    /* 2. Impact cell has a distance value of 1 */
    distance = distance + 1;

    /* 3. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 4. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 5. Do not add if its absolute value is lower than the threshold */
    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
        layer[k] = layer[k] + energy_k;
}



////////////////////////////// Naive Memory Implementations ///////////////////////////////////////////////////////////////

__global__
void bombardment_kernel(float *d_layer, int layer_sz, int *d_particles, int particles_sz){
    // Each thread is responsible of a layer cell  
    int layer_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (layer_idx >= layer_sz) return; // boundary check

    // Update the layer cell for all particles in the storm

    for (int i = 0; i < particles_sz; i++){
        float energy = (float)d_particles[i*2 + 1] * 1000; // energy in thousandths of Joule
        int position = d_particles[i*2]; // impact position

        update(d_layer, layer_sz, layer_idx, position, energy);
    }
}

__global__
void relaxation_kernel(float *d_layer_copy, float *d_layer, int layer_sz){
    // Each thread must update the output layer vector by taking into account
    // itself, neighbour -1 and neighbour +1.
    int layer_idx= threadIdx.x + blockIdx.x * blockDim.x;
    if (layer_idx <= 0 || layer_idx >= layer_sz-1) {
        d_layer[layer_idx] = d_layer_copy[layer_idx]; // first and last should be saved as they are 
        return;
    }

    d_layer[layer_idx] = (d_layer_copy[layer_idx-1] + d_layer_copy[layer_idx] +  d_layer_copy[layer_idx+1]) / 3;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

////////////////////////////// Shared Memory implementations ///////////////////////////////////////////////////////////////

__global__
void bombardment_kernelsh(){
}

__global__
void relaxation_kernelsh(){
    // load left / right ghost values from shared memory 
}

__global__
void maxval_kernelsh(int layer_sz){
   
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, k;
    float *h_layer = (float *)malloc( sizeof(float) * layer_size );
    if (h_layer == NULL) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }

    float *d_layer;  float *d_layer_copy;
    int *d_particles; // particles in a wave 

    CUDA_CHECK(cudaMalloc(&d_layer, sizeof(float) * layer_size)); // one layer for the whole execution
    CUDA_CHECK(cudaMalloc(&d_layer_copy, sizeof(float) * layer_size));
    // CUDA_CHECK(cudaMemcpy(d_layer, h_layer, sizeof(float) * layer_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_layer, 0, sizeof(float) * layer_size)); // instead of memcpy of zeros 

    dim3 block(BLOCKSIZE);
    dim3 grid((layer_size + BLOCKSIZE -1) / BLOCKSIZE);

    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {
        //////////////////////////////////////////////////////////////////
        // Bombardment Phase
        CUDA_CHECK(cudaMalloc(&d_particles, sizeof(int) * storms[i].size * 2)); // one d_particles per wave (storm) 
        CUDA_CHECK(cudaMemcpy(d_particles, storms[i].posval, sizeof(int) * storms[i].size * 2, cudaMemcpyHostToDevice));

        bombardment_kernel<<<grid, block>>>(d_layer, layer_size, d_particles, storms[i].size);
        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaDeviceSynchronize());  
        //////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////
        // Relaxation Phase
        CUDA_CHECK(cudaMemcpy(d_layer_copy, d_layer, sizeof(float) * layer_size, cudaMemcpyDeviceToDevice));

        relaxation_kernel<<<grid, block>>>(d_layer_copy, d_layer, layer_size);
        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaDeviceSynchronize());  

        CUDA_CHECK(cudaMemcpy(h_layer, d_layer, sizeof(float) * layer_size, cudaMemcpyDeviceToHost));  // todo remove
        //////////////////////////////////////////////////////////////////


        //////////////////////////////////////////////////////////////////
        // Maximum Value Location Phase 
        /* 4.3. Locate the maximum value in the layer, and its position */
        for(k=1; k<layer_size-1; k++) {
            /* Check it only if it is a local maximum */
            if ( h_layer[k] > h_layer[k-1] && h_layer[k] > h_layer[k+1] ) {
                if ( h_layer[k] > maximum[i] ) {
                    maximum[i] = h_layer[k];
                    positions[i] = k;
                }
            }
        }
        //////////////////////////////////////////////////////////////////


        //////////////////////////////////////////////////////////////////
        // Cleanup
        CUDA_CHECK(cudaFree(d_particles));
    }

    CUDA_CHECK(cudaFree(d_layer));
}