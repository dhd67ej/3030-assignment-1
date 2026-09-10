/*
 * File:     part1b.c
 *
 * Purpose:
 *    Assignment Part 1B - Reduced-Memory MPI N-body Design
 *
 *    This program is based on the supplied mpi_nbody_basic.c.
 *
 *    Part 1A:
 *       Every process permanently stored the complete position
 *       array pos[n], and ring communication reconstructed the
 *       global position state after each timestep.
 *
 *    Part 1B:
 *       Every process permanently stores ONLY:
 *
 *          - masses of its own loc_n particles
 *          - positions of its own loc_n particles
 *          - velocities of its own loc_n particles
 *          - forces on its own loc_n particles
 *
 *       Remote particle mass/position data is held only
 *       temporarily in ring communication buffers.
 *
 *       During each timestep:
 *
 *          1. Zero local forces.
 *          2. Calculate force contribution from the local block.
 *          3. Send a mass+position block to the right neighbour.
 *          4. Receive another block from the left neighbour.
 *          5. Immediately accumulate its force contribution.
 *          6. Forward the received block around the ring.
 *          7. After all p blocks have been processed, update only
 *             locally owned positions and velocities.
 *
 *    MPI_Sendrecv is used for deadlock-free point-to-point
 *    communication between ring neighbours.
 *
 *
 * Compile:
 *
 *    mpicc -g -Wall -o part1b part1b.c -lm
 *
 *
 * Run:
 *
 *    mpiexec -n <processes> ./part1b
 *       <particles> <timesteps> <delta_t> <output_freq> <g|i>
 *
 *
 * Example:
 *
 *    mpiexec -n 2 ./part1b 4 10 0.01 1 i < input_n4
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


#define DIM 2
#define X 0
#define Y 1


typedef double vect_t[DIM];


/*
 * A ring block contains exactly the information needed
 * to calculate gravitational force from a remote particle:
 *
 *     mass
 *     x position
 *     y position
 *
 * Velocity does NOT need to travel around the ring.
 */
typedef struct {
   double mass;
   double pos[DIM];
} body_block_t;


/*--------------------------------------------------------------------*/
/* Global MPI variables                                               */
/*--------------------------------------------------------------------*/

const double G = 6.673e-11;

int my_rank;
int comm_sz;

MPI_Comm comm;

MPI_Datatype vect_mpi_t;
MPI_Datatype body_block_mpi_t;


/*--------------------------------------------------------------------*/
/* Function declarations                                              */
/*--------------------------------------------------------------------*/

void Usage(char *prog_name);

void Get_args(
      int argc,
      char *argv[],
      int *n_p,
      int *n_steps_p,
      double *delta_t_p,
      int *output_freq_p,
      char *g_i_p
);

void Get_init_cond(
      double loc_masses[],
      vect_t loc_pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
);

void Gen_init_cond(
      double loc_masses[],
      vect_t loc_pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
);

void Output_state(
      double time,
      vect_t loc_pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
);

void Accumulate_block_forces(
      double loc_masses[],
      vect_t loc_pos[],
      vect_t loc_forces[],
      body_block_t block[],
      int block_owner,
      int loc_n
);

void Compute_forces_ring(
      double loc_masses[],
      vect_t loc_pos[],
      vect_t loc_forces[],
      body_block_t send_block[],
      body_block_t recv_block[],
      int loc_n
);

void Update_part(
      int loc_part,
      double loc_masses[],
      vect_t loc_forces[],
      vect_t loc_pos[],
      vect_t loc_vel[],
      double delta_t
);


/*====================================================================*/
/* MAIN                                                               */
/*====================================================================*/

int main(int argc, char *argv[]) {

   int n;
   int loc_n;

   int n_steps;
   int step;

   int loc_part;
   int output_freq;

   double delta_t;
   double t;

   char g_i;

   double start;
   double finish;


   /*
    * Permanent per-process data.
    *
    * Note carefully:
    *
    * There is NO:
    *
    *     masses[n]
    *     pos[n]
    *
    * on every rank.
    *
    * Each rank stores only loc_n values.
    */
   double *loc_masses;

   vect_t *loc_pos;
   vect_t *loc_vel;
   vect_t *loc_forces;


   /*
    * Temporary communication buffers.
    *
    * Each buffer contains only loc_n particles.
    */
   body_block_t *send_block;
   body_block_t *recv_block;


   /*---------------------------------------------------------------
    * Initialise MPI
    *---------------------------------------------------------------*/

   MPI_Init(&argc, &argv);

   comm = MPI_COMM_WORLD;

   MPI_Comm_size(
         comm,
         &comm_sz
   );

   MPI_Comm_rank(
         comm,
         &my_rank
   );


   /*---------------------------------------------------------------
    * Command-line arguments
    *---------------------------------------------------------------*/

   Get_args(
         argc,
         argv,
         &n,
         &n_steps,
         &delta_t,
         &output_freq,
         &g_i
   );


   if (n % comm_sz != 0) {

      if (my_rank == 0) {

         fprintf(
               stderr,
               "Error: number of particles must be evenly "
               "divisible by number of MPI processes.\n"
         );
      }

      MPI_Finalize();

      return 1;
   }


   loc_n =
         n / comm_sz;


   /*---------------------------------------------------------------
    * Allocate ONLY locally owned permanent state
    *---------------------------------------------------------------*/

   loc_masses =
         malloc(
               loc_n * sizeof(double)
         );


   loc_pos =
         malloc(
               loc_n * sizeof(vect_t)
         );


   loc_vel =
         malloc(
               loc_n * sizeof(vect_t)
         );


   loc_forces =
         malloc(
               loc_n * sizeof(vect_t)
         );


   /*
    * Two temporary ring buffers.
    */
   send_block =
         malloc(
               loc_n * sizeof(body_block_t)
         );


   recv_block =
         malloc(
               loc_n * sizeof(body_block_t)
         );


   if (
         loc_masses == NULL ||
         loc_pos == NULL ||
         loc_vel == NULL ||
         loc_forces == NULL ||
         send_block == NULL ||
         recv_block == NULL
      ) {

      fprintf(
            stderr,
            "Process %d: memory allocation failed\n",
            my_rank
      );

      MPI_Abort(
            comm,
            EXIT_FAILURE
      );
   }


   /*---------------------------------------------------------------
    * MPI datatype for vect_t
    *---------------------------------------------------------------*/

   MPI_Type_contiguous(
         DIM,
         MPI_DOUBLE,
         &vect_mpi_t
   );

   MPI_Type_commit(
         &vect_mpi_t
   );


   /*
    * body_block_t consists of:
    *
    *     mass
    *     x
    *     y
    *
    * which are three consecutive doubles.
    */
   MPI_Type_contiguous(
         3,
         MPI_DOUBLE,
         &body_block_mpi_t
   );

   MPI_Type_commit(
         &body_block_mpi_t
   );


   /*---------------------------------------------------------------
    * Initial conditions
    *---------------------------------------------------------------*/

   if (g_i == 'i') {

      Get_init_cond(
            loc_masses,
            loc_pos,
            loc_vel,
            n,
            loc_n
      );

   } else {

      Gen_init_cond(
            loc_masses,
            loc_pos,
            loc_vel,
            n,
            loc_n
      );
   }


   start =
         MPI_Wtime();


#ifndef NO_OUTPUT

   /*
    * Preserve exactly the same state-snapshot format
    * as the supplied starter code.
    */
   Output_state(
         0.0,
         loc_pos,
         loc_vel,
         n,
         loc_n
   );

#endif


   /*================================================================
    * MAIN TIMESTEP LOOP
    *================================================================*/

   for (
         step = 1;
         step <= n_steps;
         step++
       ) {

      t =
            step * delta_t;


      /*-------------------------------------------------------------
       * Zero local force accumulators
       *-------------------------------------------------------------*/

      for (
            loc_part = 0;
            loc_part < loc_n;
            loc_part++
          ) {

         loc_forces[loc_part][X] =
               0.0;

         loc_forces[loc_part][Y] =
               0.0;
      }


      /*-------------------------------------------------------------
       * PART 1B:
       *
       * Accumulate forces while mass/position blocks travel
       * around the ring.
       *
       * No complete global mass or position arrays are built.
       *-------------------------------------------------------------*/

      Compute_forces_ring(
            loc_masses,
            loc_pos,
            loc_forces,
            send_block,
            recv_block,
            loc_n
      );


      /*-------------------------------------------------------------
       * After ALL p blocks have contributed to the force,
       * update only the bodies owned by this rank.
       *-------------------------------------------------------------*/

      for (
            loc_part = 0;
            loc_part < loc_n;
            loc_part++
          ) {

         Update_part(
               loc_part,
               loc_masses,
               loc_forces,
               loc_pos,
               loc_vel,
               delta_t
         );
      }


#ifndef NO_OUTPUT

      if (
            step % output_freq == 0
         ) {

         Output_state(
               t,
               loc_pos,
               loc_vel,
               n,
               loc_n
         );
      }

#endif

   }


   finish =
         MPI_Wtime();


   if (
         my_rank == 0
      ) {

      printf(
            "Elapsed time = %e seconds\n",
            finish - start
      );
   }


   /*---------------------------------------------------------------
    * Cleanup
    *---------------------------------------------------------------*/

   MPI_Type_free(
         &body_block_mpi_t
   );


   MPI_Type_free(
         &vect_mpi_t
   );


   free(
         loc_masses
   );


   free(
         loc_pos
   );


   free(
         loc_vel
   );


   free(
         loc_forces
   );


   free(
         send_block
   );


   free(
         recv_block
   );


   MPI_Finalize();

   return 0;
}


/*====================================================================*/
/* COMPUTE FORCES USING REDUCED-MEMORY RING                           */
/*====================================================================*/

void Compute_forces_ring(
      double loc_masses[],
      vect_t loc_pos[],
      vect_t loc_forces[],
      body_block_t send_block[],
      body_block_t recv_block[],
      int loc_n
) {

   int stage;
   int i;

   int left;
   int right;

   int block_owner;


   /*---------------------------------------------------------------
    * Ring neighbours
    *---------------------------------------------------------------*/

   left =
         (my_rank - 1 + comm_sz)
         % comm_sz;


   right =
         (my_rank + 1)
         % comm_sz;


   /*---------------------------------------------------------------
    * Build the first communication block from the locally
    * owned mass and position data.
    *---------------------------------------------------------------*/

   for (
         i = 0;
         i < loc_n;
         i++
       ) {

      send_block[i].mass =
            loc_masses[i];


      send_block[i].pos[X] =
            loc_pos[i][X];


      send_block[i].pos[Y] =
            loc_pos[i][Y];
   }


   /*
    * Initially the block belongs to this rank.
    */
   block_owner =
         my_rank;


   /*---------------------------------------------------------------
    * Each process must process exactly comm_sz blocks:
    *
    *     1 local block
    *     comm_sz - 1 remote blocks
    *
    * Example p = 4:
    *
    * Rank 0 sees:
    *
    *     block 0
    *     block 3
    *     block 2
    *     block 1
    *
    *---------------------------------------------------------------*/

   for (
         stage = 0;
         stage < comm_sz;
         stage++
       ) {


      /*
       * Immediately use the block currently held by this rank.
       *
       * We DO NOT save the remote block permanently.
       */
      Accumulate_block_forces(
            loc_masses,
            loc_pos,
            loc_forces,
            send_block,
            block_owner,
            loc_n
      );


      /*
       * After processing the final block,
       * no further communication is necessary.
       */
      if (
            stage == comm_sz - 1
         ) {

         break;
      }


#ifdef DEBUG

      printf(
            "Proc %d > ring stage %d: "
            "send block %d to %d, receive from %d\n",
            my_rank,
            stage + 1,
            block_owner,
            right,
            left
      );

#endif


      /*------------------------------------------------------------
       * Point-to-point ring communication.
       *
       * Send:
       *     one loc_n block of mass + position
       *
       * Receive:
       *     one loc_n block of mass + position
       *
       * MPI_Sendrecv avoids the blocking-send circular deadlock.
       *------------------------------------------------------------*/

      MPI_Sendrecv(
            send_block,
            loc_n,
            body_block_mpi_t,
            right,
            0,

            recv_block,
            loc_n,
            body_block_mpi_t,
            left,
            0,

            comm,
            MPI_STATUS_IGNORE
      );


      /*
       * The block received from the left originated
       * one rank earlier around the ring.
       */
      block_owner =
            (block_owner - 1 + comm_sz)
            % comm_sz;


      /*
       * Reuse buffers.
       *
       * Instead of copying loc_n structures, simply exchange
       * the two pointers.
       *
       * The newly received block becomes the block that will
       * be processed and forwarded in the next stage.
       */
      body_block_t *tmp =
            send_block;

      send_block =
            recv_block;

      recv_block =
            tmp;
   }
}


/*====================================================================*/
/* ACCUMULATE FORCE FROM ONE MASS/POSITION BLOCK                       */
/*====================================================================*/

void Accumulate_block_forces(
      double loc_masses[],
      vect_t loc_pos[],
      vect_t loc_forces[],
      body_block_t block[],
      int block_owner,
      int loc_n
) {

   int loc_part;
   int k;

   double mg;

   vect_t f_part_k;

   double len;
   double len_3;
   double fact;


   /*
    * Calculate contribution of every particle in this
    * ring block to every locally owned particle.
    */
   for (
         loc_part = 0;
         loc_part < loc_n;
         loc_part++
       ) {


      for (
            k = 0;
            k < loc_n;
            k++
          ) {


         /*
          * Self interaction occurs only when the block
          * currently being processed is this rank's own
          * block AND the local indices are the same.
          */
         if (
               block_owner == my_rank &&
               k == loc_part
            ) {

            continue;
         }


         /*
          * Exactly the same force calculation as the
          * supplied mpi_nbody_basic.c.
          *
          * f_part_k =
          *
          *     local_position - other_position
          */
         f_part_k[X] =
               loc_pos[loc_part][X]
               -
               block[k].pos[X];


         f_part_k[Y] =
               loc_pos[loc_part][Y]
               -
               block[k].pos[Y];


         len =
               sqrt(
                     f_part_k[X] *
                     f_part_k[X]
                     +
                     f_part_k[Y] *
                     f_part_k[Y]
               );


         len_3 =
               len *
               len *
               len;


         mg =
               -G *
               loc_masses[loc_part] *
               block[k].mass;


         fact =
               mg /
               len_3;


         f_part_k[X] *=
               fact;


         f_part_k[Y] *=
               fact;


         loc_forces[loc_part][X] +=
               f_part_k[X];


         loc_forces[loc_part][Y] +=
               f_part_k[Y];


#ifdef DEBUG

         printf(
               "Proc %d > block %d: force on local part %d "
               "due to block part %d = (%.3e, %.3e)\n",
               my_rank,
               block_owner,
               loc_part,
               k,
               f_part_k[X],
               f_part_k[Y]
         );

#endif
      }
   }
}


/*====================================================================*/
/* UPDATE LOCAL PARTICLE                                              */
/*====================================================================*/

void Update_part(
      int loc_part,
      double loc_masses[],
      vect_t loc_forces[],
      vect_t loc_pos[],
      vect_t loc_vel[],
      double delta_t
) {

   double fact;


   fact =
         delta_t
         /
         loc_masses[loc_part];


   /*
    * IMPORTANT:
    *
    * Preserve exactly the Euler update order of
    * mpi_nbody_basic.c.
    *
    * Position is updated using the OLD velocity first.
    *
    * Then velocity is updated using force.
    */


   loc_pos[loc_part][X] +=
         delta_t *
         loc_vel[loc_part][X];


   loc_pos[loc_part][Y] +=
         delta_t *
         loc_vel[loc_part][Y];


   loc_vel[loc_part][X] +=
         fact *
         loc_forces[loc_part][X];


   loc_vel[loc_part][Y] +=
         fact *
         loc_forces[loc_part][Y];
}


/*====================================================================*/
/* OUTPUT                                                             */
/*====================================================================*/

void Output_state(
      double time,
      vect_t loc_pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
) {

   int part;


   /*
    * These global arrays exist ONLY temporarily on rank 0
    * while an output snapshot is being produced.
    *
    * They are NOT permanent simulation state and are
    * immediately freed after printing.
    */
   vect_t *out_pos =
         NULL;


   vect_t *out_vel =
         NULL;


   if (
         my_rank == 0
      ) {

      out_pos =
            malloc(
                  n * sizeof(vect_t)
            );


      out_vel =
            malloc(
                  n * sizeof(vect_t)
            );


      if (
            out_pos == NULL ||
            out_vel == NULL
         ) {

         fprintf(
               stderr,
               "Unable to allocate temporary output arrays\n"
         );

         MPI_Abort(
               comm,
               EXIT_FAILURE
         );
      }
   }


   /*
    * Gather local positions ONLY for output.
    *
    * This is not used for force calculation and does not
    * reconstruct global state on every MPI process.
    */
   MPI_Gather(
         loc_pos,
         loc_n,
         vect_mpi_t,

         out_pos,
         loc_n,
         vect_mpi_t,

         0,
         comm
   );


   /*
    * Gather local velocities ONLY for output.
    */
   MPI_Gather(
         loc_vel,
         loc_n,
         vect_mpi_t,

         out_vel,
         loc_n,
         vect_mpi_t,

         0,
         comm
   );


   /*
    * Preserve the supplied starter program's exact
    * state-snapshot output format.
    */
   if (
         my_rank == 0
      ) {

      printf(
            "%.2f\n",
            time
      );


      for (
            part = 0;
            part < n;
            part++
          ) {

         printf(
               "%3d %10.3e ",
               part,
               out_pos[part][X]
         );


         printf(
               "  %10.3e ",
               out_pos[part][Y]
         );


         printf(
               "  %10.3e ",
               out_vel[part][X]
         );


         printf(
               "  %10.3e\n",
               out_vel[part][Y]
         );
      }


      printf(
            "\n"
      );


      /*
       * The full arrays are temporary only.
       */
      free(
            out_pos
      );


      free(
            out_vel
      );
   }
}


/*====================================================================*/
/* READ INITIAL CONDITIONS                                            */
/*====================================================================*/

void Get_init_cond(
      double loc_masses[],
      vect_t loc_pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
) {

   int part;


   /*
    * Full initial arrays exist temporarily on process 0
    * because input arrives through stdin in global particle order.
    *
    * They are scattered immediately and then freed.
    *
    * Therefore they are NOT part of the permanent memory
    * requirements of the simulation.
    */
   double *temp_masses =
         NULL;


   vect_t *temp_pos =
         NULL;


   vect_t *temp_vel =
         NULL;


   if (
         my_rank == 0
      ) {

      temp_masses =
            malloc(
                  n * sizeof(double)
            );


      temp_pos =
            malloc(
                  n * sizeof(vect_t)
            );


      temp_vel =
            malloc(
                  n * sizeof(vect_t)
            );


      if (
            temp_masses == NULL ||
            temp_pos == NULL ||
            temp_vel == NULL
         ) {

         fprintf(
               stderr,
               "Unable to allocate temporary input arrays\n"
         );

         MPI_Abort(
               comm,
               EXIT_FAILURE
         );
      }


      /*
       * Preserve the exact input prompt from
       * mpi_nbody_basic.c.
       */
      printf(
            "For each particle, enter (in order):\n"
      );


      printf(
            "   its mass, its x-coord, its y-coord, "
      );


      printf(
            "its x-velocity, its y-velocity\n"
      );


      for (
            part = 0;
            part < n;
            part++
          ) {

         scanf(
               "%lf",
               &temp_masses[part]
         );


         scanf(
               "%lf",
               &temp_pos[part][X]
         );


         scanf(
               "%lf",
               &temp_pos[part][Y]
         );


         scanf(
               "%lf",
               &temp_vel[part][X]
         );


         scanf(
               "%lf",
               &temp_vel[part][Y]
         );
      }
   }


   /*
    * Unlike mpi_nbody_basic.c, masses and positions
    * are NOT broadcast globally.
    *
    * Each process receives only its own block.
    */
   MPI_Scatter(
         temp_masses,
         loc_n,
         MPI_DOUBLE,

         loc_masses,
         loc_n,
         MPI_DOUBLE,

         0,
         comm
   );


   MPI_Scatter(
         temp_pos,
         loc_n,
         vect_mpi_t,

         loc_pos,
         loc_n,
         vect_mpi_t,

         0,
         comm
   );


   MPI_Scatter(
         temp_vel,
         loc_n,
         vect_mpi_t,

         loc_vel,
         loc_n,
         vect_mpi_t,

         0,
         comm
   );


   if (
         my_rank == 0
      ) {

      free(
            temp_masses
      );


      free(
            temp_pos
      );


      free(
            temp_vel
      );
   }
}


/*====================================================================*/
/* GENERATE INITIAL CONDITIONS                                        */
/*====================================================================*/

void Gen_init_cond(
      double loc_masses[],
      vect_t loc_pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
) {

   int loc_part;
   int part;

   double mass =
         5.0e24;


   double gap =
         1.0e5;


   double speed =
         3.0e4;


   /*
    * No global arrays are required here.
    *
    * Since the generated initial conditions are
    * deterministic, each process can generate exactly
    * the particles that it owns.
    */
   for (
         loc_part = 0;
         loc_part < loc_n;
         loc_part++
       ) {


      /*
       * Global particle index.
       */
      part =
            my_rank *
            loc_n
            +
            loc_part;


      loc_masses[loc_part] =
            mass;


      loc_pos[loc_part][X] =
            part *
            gap;


      loc_pos[loc_part][Y] =
            0.0;


      loc_vel[loc_part][X] =
            0.0;


      if (
            part % 2 == 0
         ) {

         loc_vel[loc_part][Y] =
               speed;

      } else {

         loc_vel[loc_part][Y] =
               -speed;
      }
   }


   /*
    * n is intentionally retained in the interface
    * for consistency.
    */
   (void)n;
}


/*====================================================================*/
/* COMMAND-LINE ARGUMENTS                                             */
/*====================================================================*/

void Get_args(
      int argc,
      char *argv[],
      int *n_p,
      int *n_steps_p,
      double *delta_t_p,
      int *output_freq_p,
      char *g_i_p
) {

   if (
         my_rank == 0
      ) {


      if (
            argc != 6
         ) {

         Usage(
               argv[0]
         );
      }


      *n_p =
            strtol(
                  argv[1],
                  NULL,
                  10
            );


      *n_steps_p =
            strtol(
                  argv[2],
                  NULL,
                  10
            );


      *delta_t_p =
            strtod(
                  argv[3],
                  NULL
            );


      *output_freq_p =
            strtol(
                  argv[4],
                  NULL,
                  10
            );


      *g_i_p =
            argv[5][0];
   }


   MPI_Bcast(
         n_p,
         1,
         MPI_INT,
         0,
         comm
   );


   MPI_Bcast(
         n_steps_p,
         1,
         MPI_INT,
         0,
         comm
   );


   MPI_Bcast(
         delta_t_p,
         1,
         MPI_DOUBLE,
         0,
         comm
   );


   MPI_Bcast(
         output_freq_p,
         1,
         MPI_INT,
         0,
         comm
   );


   MPI_Bcast(
         g_i_p,
         1,
         MPI_CHAR,
         0,
         comm
   );


   if (
         *n_p <= 0 ||
         *n_steps_p < 0 ||
         *delta_t_p <= 0 ||
         *output_freq_p <= 0
      ) {


      if (
            my_rank == 0
         ) {

         Usage(
               argv[0]
         );
      }


      MPI_Finalize();

      exit(0);
   }


   if (
         *g_i_p != 'g' &&
         *g_i_p != 'i'
      ) {


      if (
            my_rank == 0
         ) {

         Usage(
               argv[0]
         );
      }


      MPI_Finalize();

      exit(0);
   }


#ifdef DEBUG

   if (
         my_rank == 0
      ) {

      printf(
            "n = %d\n",
            *n_p
      );


      printf(
            "n_steps = %d\n",
            *n_steps_p
      );


      printf(
            "delta_t = %e\n",
            *delta_t_p
      );


      printf(
            "output_freq = %d\n",
            *output_freq_p
      );


      printf(
            "g_i = %c\n",
            *g_i_p
      );
   }

#endif
}


/*====================================================================*/
/* USAGE                                                              */
/*====================================================================*/

void Usage(
      char *prog_name
) {

   fprintf(
         stderr,
         "usage: mpiexec -n <number of processes> %s\n",
         prog_name
   );


   fprintf(
         stderr,
         "   <number of particles> <number of timesteps>\n"
   );


   fprintf(
         stderr,
         "   <size of timestep> <output frequency>\n"
   );


   fprintf(
         stderr,
         "   <g|i>\n"
   );


   fprintf(
         stderr,
         "   'g': program should generate init conds\n"
   );


   fprintf(
         stderr,
         "   'i': program should get init conds from stdin\n"
   );


   exit(0);
}
