#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "energy_storms.h"
#include "mpi.h"
#include "omp.h"

#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define ZERO_OR_ONE(x,y) (((x) < (y)) ? (1) : (0))
#define MAX_DEBUG_SIZE 35
/* ------------------------------------------------------------ Data ------------------------------------------------------------> */
struct reductionResult{
    float val;
    int pos;
};

/* ------------------------------------------------------------ Global ----------------------------------------------------------> */
/* This global vector, is just used to join the local_layer.
 * Is used in the main function, to call the "print_debug" function, in the "energy_storms.h" */
float *layer = NULL;

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer
 * */
static float updateControlPoint( float *local_layer, int layer_size, int k, int pos, float energy ) {
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
        return energy_k;
    return 0.0f;
}


void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    /* Let's define alse here the global comunicator and the rank variables */
    int rank, comm_sz;
    int i, j, k;
    double t_start, t_comp, t_comm, t_finish, local_elapsed, elapsed;
    struct reductionResult localResult, globalResult;


    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* To devide the array size proportionaly with the number of process.
     * Evaluate alse the correct number of index fot that proces */
    int sub_domain = layer_size / comm_sz;
    int rest = layer_size % comm_sz;
    int local_start = rank * sub_domain + MIN(rank, rest);
    int local_size = sub_domain + ZERO_OR_ONE(rank, rest);
    int local_end = local_size + local_start;

    /* 3. Allocate memory for the layer and initialize to zero
     *  for this allocation in mamory we are gonna teke into account 2 hidden position border
     * */

    float *local_layer = (float *)calloc(local_size + 2, sizeof(float));
    float *local_layer_copy = (float *)calloc(local_size + 2, sizeof(float));
    if ( local_layer == NULL || local_layer_copy == NULL ) {
        fprintf(stderr,"Error: Allocating the local layer memory\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit( EXIT_FAILURE );
    }
    int *displs = NULL;
    int *recvcounts = NULL;

/* Let's define here the necessary for DDEBUG part of the exercise
 * - Define all the ancillary array for the "MPI_GATHERV" operation:
 * - recvcounts: array of metadata, used to save how many space is needed to for each data coming, from the i-th rank
 * - displ: array of metadata, used to save the exact position in whitch the data beginning in the join array result
 **/
#ifdef DDEBUG
        /* Allocation of metadata array*/
        recvcounts = malloc(sizeof(int) * comm_sz);
        displs = malloc(sizeof(int) * comm_sz);
        layer = malloc(sizeof(float) * layer_size);

        if (recvcounts == NULL || displs == NULL || layer == NULL){
            fprintf(stderr,"Error: Allocating the local layer memory\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            exit( EXIT_FAILURE );
        }

        /* Too small the loop to parallelize with OMP could be much expensive */
        int offset = 0;
        for (i = 0; i<comm_sz; i++){
            int length = (layer_size / comm_sz) + ZERO_OR_ONE(i, rest);
            recvcounts[i] = length;
            displs[i] = offset;
            /* "offset" variable in need to update the display in a correct way and avoid to reed a null value fro it's self */
            offset += length;
        }
#endif
        /* "local_layer" and "local_layer_copy" initialization.
         *  Here we itialize to '0.0f'the sub-array and it's ancillary array.
         *  We are taking in to account also the space of '2' in addition, for the halo exchange*/
        #pragma omp parallel for schedule(static) default(none) private(k) shared(local_size, local_layer, local_layer_copy)
        for (k=0; k<local_size+2; k++){
            local_layer[k] = 0.0f;
            local_layer_copy[k] = 0.0f;
        }

        int max_size = 0;
        /* Here we defined a loop for find the biggest 'storm[i].size'.
         * This data is used to allocate the space for the 's_pos' and 's_en'*/
        #pragma omp parallel for schedule(static) reduction(max: max_size)
        for (i=0; i<num_storms; i++){
            if (storms[i].size > max_size)
                max_size = storms[i].size;
        }
         /* This two guy are used to same the 'energy' and 'position' of the particel, inside of the sequential were computed at each loop.
          * During the OMP implementation this two array were itiliazated at each 'j' loop.
          * For performance reason is been initializated just one time, for the maximum space needed
          * This is also a profiling best practice*/
        float *s_pos = malloc(max_size * sizeof(float));
        float *s_en = malloc(max_size * sizeof(float));
        if (s_pos == NULL || s_en == NULL){
               fprintf(stderr,"Error: Allocating the local layer memory\n");
               MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
               exit( EXIT_FAILURE );
        }

        /* Variable to save a redundant operation, evaluated just one time */
        float seq_threshold = THRESHOLD / (float)layer_size;
        t_start = MPI_Wtime();

        /* Main loop */
        for(i = 0; i < num_storms; i++) {
            int s_size = storms[i].size;

            /* Data Paking, colelct the information about 'energy' and 'position'*/
            #pragma omp parallel for schedule(static)
            for(int p = 0; p < s_size; p++) {
                s_pos[p] = (float)storms[i].posval[p*2];
                s_en[p] = (float)storms[i].posval[p*2+1] * 1000.0f;
            }

            /*---------------------------------------------------------------------------- Paralle Loop -------------------------------------------------------------------- */
            /* This is the haviest loop insied of the code, here there is all the mathematic calcolus.
             * Initialt in the sequential code, all this part was in the 'update()' function but for performance reason we save time avoiding useless functions call.
             * At the beginnig of parallel code the local_layer was updated using an atomic directive, becoming a bottle neck for the performance.
             * To improving a litle bit the performance exchange the loops, so in this way change the owner of the memory in that moment, before the parallelization occurred on the jth particel,
             * Since differnt particels (handled by different threads) can hit the same positio k, there was needed "#pragma omp atomic"
             *
             * */
            t_comp = MPI_Wtime();
            #pragma omp parallel for schedule(runtime)
            for (int k = 1; k <= local_size; k++) {
                float global_id = (float)(local_start + k - 1);

                for (int j = 0; j < s_size; j++) {
                    float distance = fabsf(s_pos[j] - global_id);

                    // VERIFICA QUESTA FORMULA CON IL SEQUENZIALE!
                    float atenuacion = sqrtf(distance + 1.0f);
                    float energy_k = s_en[j] / (float)layer_size / atenuacion;

                    if (energy_k >= seq_threshold || energy_k <= -seq_threshold) {
                        local_layer[k] += energy_k;
                    }
                }
            }


             /*  We decide to use Sendrecv collective, to leverage it's self handling of send/recv
             *  Here we trat the halo exchange problem. Exchange the halo data between process
             *  MPI_PROC_NULL is used to ignore the send or recv of sender, to keep invariant the respected buffers,
             *  Without the MPI_PROC_NULL we should add a loot of conditional if, to handle potential errors
             *
             * */
            int req_count=0;
            /* Here we set request just to 4, because each rank will do exactly this number of point-to-point calls  */
            t_start = MPI_Wtime();
            MPI_Request request[4];

            if (rank > 0){
                MPI_Isend(&local_layer[1], 1, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &request[req_count++]);
                MPI_Irecv(&local_layer[0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &request[req_count++]);

            }if (rank < comm_sz - 1){
                MPI_Isend(&local_layer[local_size], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &request[req_count++]);
                MPI_Irecv(&local_layer[local_size +1], 1, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &request[req_count++]);
            }

            /* MPI_Waitall: lets you to wait all the point - point comunication
             * It's necessary before to use the sanded data*/

            MPI_Waitall(req_count, request, MPI_STATUSES_IGNORE);
            t_finish = MPI_Wtime();
            local_elapsed = t_finish - t_start;
#if DIS_PROFILING == true
            MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX ,0, MPI_COMM_WORLD);
            if(rank == 0) printf("Time for ending the halo exchange: %f\n", elapsed);
#endif
            /* Send the first right border element to the gosth cell  */
            /*
             * MPI_Sendrecv(&local_layer[1], 1, MPI_FLOAT, left_neighbor, 0, &local_layer[local_size + 1], 1, MPI_FLOAT, right_neighbor, 0, MPI_COMM_WORLD, &status);
               MPI_Sendrecv(&local_layer[local_size], 1, MPI_FLOAT, right_neighbor, 0, &local_layer[0], 1, MPI_FLOAT, left_neighbor, 0, MPI_COMM_WORLD, &status);
            */


            /* 4.2. Energy relaxation between storms */
            /* 4.2.1. Copy values to the ancillary array */
            /* The global_max_pos, save the position of the global array, while the local_max, save the
             * local maximum value for the current sub array*/
            float local_max = -FLT_MAX;
            int  global_max_pos= -1;
            /* Here we create the thread team, but wee keep it worm, for the 'for' task allocation, always for performance reason */
            #pragma omp parallel default(none) private (k, i) shared(local_size, local_layer, local_layer_copy, maximum, positions, local_max, global_max_pos, local_start, rank)
            {
                /* 'simd' to vectorializate the operation*/
                #pragma omp for simd schedule(runtime)
                for( k=0; k<local_size+2; k++ )
                    local_layer_copy[k] = local_layer[k];

                /* 4.2.2. Update layer using the ancillary values.
                          Skip updating the first and last positions
                */

                /* Here the update of the value is done taking in account the two extra cells inside the 'local_layer' and it's ancillary array.*/
                #pragma omp for schedule(runtime)
                for( k=1; k<local_size; k++ ){
                   local_layer[k] = ( local_layer_copy[k-1] + local_layer_copy[k] + local_layer_copy[k+1] ) / 3.0f;
                }
                /*4.3. Locate the maximum value in the layer, and its position */
                /* The local maximux, of each rank, is found using OpenOM.
                 * Firtstly we parallelize the loop, and at each iteration we check if the current value can be candidated to be 'thread_max'.
                 * At the end, outside the loop, with a critical section, we get the 'local_max' and the 'global_max_pos'*/
               float thread_max = -FLT_MAX;
               int thread_max_pos = -1;
               #pragma omp for schedule(runtime) nowait
                for( k=1; k<local_size-1; k++ ) {
                    /* Check it only if it is a local maximum */
                    if ( local_layer[k] > local_layer[k-1] && local_layer[k] > local_layer[k+1] ) {
                        if ( local_layer[k] > thread_max ) {
                            thread_max = local_layer[k];
                            thread_max_pos = k;
                        }
                    }
                }
            #pragma omp critical
                {
                    if(thread_max > local_max){
                        local_max = thread_max;
                        global_max_pos = local_start + thread_max_pos - 1;
                    }
                }
            }

            localResult.val = local_max;
            localResult.pos = global_max_pos;

            /* For this reduction we decide to use MPI_FLOAT_INT to get a dedicated struct, to save result and position
             * MPI_MAXLOC: execute reduction, and get the index*/
            t_start = MPI_Wtime();
            MPI_Allreduce(&localResult, &globalResult, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
            t_finish = MPI_Wtime();
            local_elapsed = t_finish - t_start;

#if DIS_PROFILING == true
            MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if(rank ==0)printf("The time for AllRecuce: %f\n", elapsed);
#endif

            maximum[i] = globalResult.val;
            positions[i] = globalResult.pos;

            #ifdef DDEBUG
            /* For the debug i need to recompore the entire array, becasue i need to print some data result
             * If the local_layer_size division gave's rest, we do MPI_Gatherv because of rest != 0 */
            if (layer_size < MAX_DEBUG_SIZE)
                MPI_Gatherv(&local_layer[1], recvcounts[rank], MPI_FLOAT, layer, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD );
            #endif
    }
        free(local_layer);
        free(local_layer_copy);
        free(s_pos);
        free(s_en);
        if (rank == 0){
            free(displs);
            free(recvcounts);
        }
}
