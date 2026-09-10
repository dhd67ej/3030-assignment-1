/*
 * File:     part1a.c
 *
 * Purpose:
 *    Part 1A - MPI Ring Communication
 *
 *    This program is based on mpi_nbody_basic.c.
 *
 *    The original implementation synchronised updated particle
 *    positions with MPI_Allgather.  In this Part 1A version,
 *    the Allgather in the timestep loop has been replaced with
 *    point-to-point communication arranged as a ring.
 *
 *    Each process:
 *       - owns loc_n particles;
 *       - computes forces on its local particles;
 *       - updates its local positions and velocities;
 *       - sends one block of updated positions around the ring;
 *       - receives one block from its left neighbour at each stage;
 *       - after comm_sz - 1 stages, has the complete updated
 *         global position array.
 *
 *    MPI_Sendrecv is used to avoid deadlock.
 *
 * Compile:
 *    mpicc -g -Wall -o part1a part1a.c -lm
 *
 * Run:
 *    mpiexec -n <number of processes> ./part1a
 *       <number of particles>
 *       <number of timesteps>
 *       <size of timestep>
 *       <output frequency>
 *       <g|i>
 *
 * Example:
 *    mpiexec -n 2 ./part1a 4 10 0.01 1 i < input_n4
 *
 * The number of particles must be evenly divisible by the
 * number of MPI processes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>


#define DIM 2
#define X 0
#define Y 1


typedef double vect_t[DIM];


/*----------------------------------------------------------*/
/* Global variables                                         */
/*----------------------------------------------------------*/

const double G = 6.673e-11;

int my_rank;
int comm_sz;

MPI_Comm comm;
MPI_Datatype vect_mpi_t;


/*
 * Scratch array used only by process 0 when gathering
 * velocities for output.
 */
vect_t *vel = NULL;


/*----------------------------------------------------------*/
/* Function prototypes                                      */
/*----------------------------------------------------------*/

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
      double masses[],
      vect_t pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
);

void Gen_init_cond(
      double masses[],
      vect_t pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
);

void Output_state(
      double time,
      double masses[],
      vect_t pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
);

void Compute_force(
      int loc_part,
      double masses[],
      vect_t loc_forces[],
      vect_t pos[],
      int n,
      int loc_n
);

void Update_part(
      int loc_part,
      double masses[],
      vect_t loc_forces[],
      vect_t loc_pos[],
      vect_t loc_vel[],
      int n,
      int loc_n,
      double delta_t
);

void Ring_exchange_positions(
      vect_t pos[],
      int loc_n
);


/*==========================================================*/
/* MAIN                                                     */
/*==========================================================*/

int main(int argc, char *argv[]) {

   int n;
   int loc_n;

   int n_steps;
   int step;
   int loc_part;

   int output_freq;

   double delta_t;
   double t;

   double *masses;

   vect_t *loc_pos;
   vect_t *pos;

   vect_t *loc_vel;
   vect_t *loc_forces;

   char g_i;

   double start;
   double finish;


   /*-------------------------------------------------------
    * Initialise MPI
    *-------------------------------------------------------*/

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


   /*-------------------------------------------------------
    * Read and distribute command-line arguments
    *-------------------------------------------------------*/

   Get_args(
         argc,
         argv,
         &n,
         &n_steps,
         &delta_t,
         &output_freq,
         &g_i
   );


   /*
    * The assignment assumes equal block partitioning.
    *
    * Therefore n must be evenly divisible by the
    * number of MPI processes.
    */
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


   loc_n = n / comm_sz;


   /*-------------------------------------------------------
    * Allocate data
    *-------------------------------------------------------*/

   masses =
         malloc(
               n * sizeof(double)
         );

   /*
    * Part 1A still stores the full position array
    * on every process.
    *
    * Part 1B will later reduce this memory usage.
    */
   pos =
         malloc(
               n * sizeof(vect_t)
         );

   loc_forces =
         malloc(
               loc_n * sizeof(vect_t)
         );


   /*
    * Each rank's own position block is a section
    * of the global position array.
    */
   loc_pos =
         pos + my_rank * loc_n;


   loc_vel =
         malloc(
               loc_n * sizeof(vect_t)
         );


   /*
    * Process 0 needs a full velocity array only
    * when producing output.
    */
   if (my_rank == 0) {

      vel =
            malloc(
                  n * sizeof(vect_t)
            );
   }


   if (masses == NULL ||
       pos == NULL ||
       loc_forces == NULL ||
       loc_vel == NULL ||
       (my_rank == 0 && vel == NULL)) {

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


   /*-------------------------------------------------------
    * MPI datatype for a 2D vector
    *-------------------------------------------------------*/

   MPI_Type_contiguous(
         DIM,
         MPI_DOUBLE,
         &vect_mpi_t
   );

   MPI_Type_commit(
         &vect_mpi_t
   );


   /*-------------------------------------------------------
    * Initial conditions
    *-------------------------------------------------------*/

   if (g_i == 'i') {

      Get_init_cond(
            masses,
            pos,
            loc_vel,
            n,
            loc_n
      );

   } else {

      Gen_init_cond(
            masses,
            pos,
            loc_vel,
            n,
            loc_n
      );
   }


   /*-------------------------------------------------------
    * Start timing
    *-------------------------------------------------------*/

   start = MPI_Wtime();


#ifndef NO_OUTPUT

   /*
    * Preserve the exact output format from the
    * supplied mpi_nbody_basic.c program.
    */
   Output_state(
         0.0,
         masses,
         pos,
         loc_vel,
         n,
         loc_n
   );

#endif


   /*=======================================================
    * Main timestep loop
    *=======================================================*/

   for (step = 1;
        step <= n_steps;
        step++) {


      t =
            step * delta_t;


      /*----------------------------------------------------
       * Compute force on each local particle.
       *
       * At this point every process already contains
       * the complete position state from the previous
       * timestep.
       *----------------------------------------------------*/

      for (loc_part = 0;
           loc_part < loc_n;
           loc_part++) {

         Compute_force(
               loc_part,
               masses,
               loc_forces,
               pos,
               n,
               loc_n
         );
      }


      /*----------------------------------------------------
       * Update the particles owned by this rank.
       *
       * This preserves the original Euler update from
       * mpi_nbody_basic.c.
       *----------------------------------------------------*/

      for (loc_part = 0;
           loc_part < loc_n;
           loc_part++) {

         Update_part(
               loc_part,
               masses,
               loc_forces,
               loc_pos,
               loc_vel,
               n,
               loc_n,
               delta_t
         );
      }


      /*----------------------------------------------------
       * PART 1A MODIFICATION
       *
       * Original code:
       *
       * MPI_Allgather(MPI_IN_PLACE, ...)
       *
       * has been replaced by a point-to-point ring.
       *
       * Every rank sends one loc_n-sized position block
       * to its right neighbour and receives one block
       * from its left neighbour.
       *
       * After comm_sz - 1 stages every process again
       * possesses the complete updated position array.
       *----------------------------------------------------*/

      Ring_exchange_positions(
            pos,
            loc_n
      );


#ifndef NO_OUTPUT

      if (step % output_freq == 0) {

         Output_state(
               t,
               masses,
               pos,
               loc_vel,
               n,
               loc_n
         );
      }

#endif

   }


   /*-------------------------------------------------------
    * Finish timing
    *-------------------------------------------------------*/

   finish = MPI_Wtime();


   if (my_rank == 0) {

      printf(
            "Elapsed time = %e seconds\n",
            finish - start
      );
   }


   /*-------------------------------------------------------
    * Cleanup
    *-------------------------------------------------------*/

   MPI_Type_free(
         &vect_mpi_t
   );


   free(
         masses
   );

   free(
         pos
   );

   free(
         loc_forces
   );

   free(
         loc_vel
   );


   if (my_rank == 0) {

      free(
            vel
      );
   }


   MPI_Finalize();

   return 0;

}


/*==========================================================*/
/* RING POSITION EXCHANGE                                   */
/*==========================================================*/

/*
 * Function:
 *    Ring_exchange_positions
 *
 * Purpose:
 *    Replace the original MPI_Allgather position exchange
 *    with point-to-point ring communication.
 *
 *
 * Ring example with four ranks:
 *
 *       Rank 0 ---> Rank 1
 *          ^           |
 *          |           v
 *       Rank 3 <--- Rank 2
 *
 *
 * Each stage:
 *
 *       send one position block to RIGHT
 *       receive one position block from LEFT
 *
 *
 * Stage 1:
 *
 *       Rank 0 sends block 0 -> Rank 1
 *       Rank 1 sends block 1 -> Rank 2
 *       Rank 2 sends block 2 -> Rank 3
 *       Rank 3 sends block 3 -> Rank 0
 *
 *
 * Stage 2:
 *
 *       Each rank forwards the block it received
 *       during the previous stage.
 *
 *
 * After comm_sz - 1 stages every rank has received
 * every other rank's updated position block.
 *
 *
 * MPI_Sendrecv is used because it safely performs
 * the matching send and receive without introducing
 * the blocking-send circular deadlock discussed in
 * the ring communication material.
 */
void Ring_exchange_positions(
      vect_t pos[],
      int loc_n
) {

   int stage;

   int left;
   int right;

   int send_owner;
   int recv_owner;

   vect_t *send_block;
   vect_t *recv_block;


   /*
    * Immediate neighbours in the ring.
    */
   left =
         (my_rank - 1 + comm_sz)
         % comm_sz;

   right =
         (my_rank + 1)
         % comm_sz;


   /*
    * With one MPI process, there are no remote
    * position blocks to exchange.
    *
    * This also ensures the n=1, p=1 autograder
    * sanity case behaves naturally.
    */
   if (comm_sz == 1) {

      return;
   }


   /*
    * There are comm_sz - 1 remote blocks.
    *
    * At each stage we forward one block clockwise.
    */
   for (stage = 1;
        stage < comm_sz;
        stage++) {


      /*
       * Block being sent at this stage.
       *
       * stage = 1:
       *    send our own block
       *
       * stage = 2:
       *    send the block received from the left
       *
       * etc.
       */
      send_owner =
            (my_rank - (stage - 1) + comm_sz)
            % comm_sz;


      /*
       * Block expected from our left neighbour.
       */
      recv_owner =
            (my_rank - stage + comm_sz)
            % comm_sz;


      /*
       * Because Part 1A keeps the complete
       * global position array, each block has a
       * permanent location inside pos[].
       */
      send_block =
            pos +
            send_owner * loc_n;


      recv_block =
            pos +
            recv_owner * loc_n;


#ifdef DEBUG

      printf(
            "Proc %d > ring stage %d: "
            "sending block %d to %d, "
            "receiving block %d from %d\n",
            my_rank,
            stage,
            send_owner,
            right,
            recv_owner,
            left
      );

#endif


      /*
       * Point-to-point neighbour communication only.
       *
       * Send one local-sized position block to the
       * right and receive one local-sized position
       * block from the left.
       *
       * MPI_Sendrecv avoids a circular wait.
       */
      MPI_Sendrecv(
            send_block,
            loc_n,
            vect_mpi_t,
            right,
            0,

            recv_block,
            loc_n,
            vect_mpi_t,
            left,
            0,

            comm,
            MPI_STATUS_IGNORE
      );
   }

}


/*==========================================================*/
/* USAGE                                                    */
/*==========================================================*/

void Usage(char *prog_name) {

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


/*==========================================================*/
/* GET COMMAND-LINE ARGUMENTS                               */
/*==========================================================*/

void Get_args(
      int argc,
      char *argv[],
      int *n_p,
      int *n_steps_p,
      double *delta_t_p,
      int *output_freq_p,
      char *g_i_p
) {

   if (my_rank == 0) {

      if (argc != 6) {

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


   if (*n_p <= 0 ||
       *n_steps_p < 0 ||
       *delta_t_p <= 0 ||
       *output_freq_p <= 0) {

      if (my_rank == 0) {

         Usage(
               argv[0]
         );
      }

      MPI_Finalize();

      exit(0);
   }


   if (*g_i_p != 'g' &&
       *g_i_p != 'i') {

      if (my_rank == 0) {

         Usage(
               argv[0]
         );
      }

      MPI_Finalize();

      exit(0);
   }


#ifdef DEBUG

   if (my_rank == 0) {

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


/*==========================================================*/
/* READ INITIAL CONDITIONS                                  */
/*==========================================================*/

void Get_init_cond(
      double masses[],
      vect_t pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
) {

   int part;


   if (my_rank == 0) {

      /*
       * Preserve the starter program's exact
       * input behaviour and output text.
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


      for (part = 0;
           part < n;
           part++) {

         scanf(
               "%lf",
               &masses[part]
         );

         scanf(
               "%lf",
               &pos[part][X]
         );

         scanf(
               "%lf",
               &pos[part][Y]
         );

         scanf(
               "%lf",
               &vel[part][X]
         );

         scanf(
               "%lf",
               &vel[part][Y]
         );
      }
   }


   /*
    * Part 1A retains the original storage model:
    * masses and initial positions are available
    * globally on each rank.
    */
   MPI_Bcast(
         masses,
         n,
         MPI_DOUBLE,
         0,
         comm
   );


   MPI_Bcast(
         pos,
         n,
         vect_mpi_t,
         0,
         comm
   );


   MPI_Scatter(
         vel,
         loc_n,
         vect_mpi_t,

         loc_vel,
         loc_n,
         vect_mpi_t,

         0,
         comm
   );

}


/*==========================================================*/
/* GENERATE INITIAL CONDITIONS                              */
/*==========================================================*/

void Gen_init_cond(
      double masses[],
      vect_t pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
) {

   int part;

   double mass =
         5.0e24;

   double gap =
         1.0e5;

   double speed =
         3.0e4;


   if (my_rank == 0) {

      for (part = 0;
           part < n;
           part++) {

         masses[part] =
               mass;


         pos[part][X] =
               part * gap;


         pos[part][Y] =
               0.0;


         vel[part][X] =
               0.0;


         if (part % 2 == 0) {

            vel[part][Y] =
                  speed;

         } else {

            vel[part][Y] =
                  -speed;
         }
      }
   }


   MPI_Bcast(
         masses,
         n,
         MPI_DOUBLE,
         0,
         comm
   );


   MPI_Bcast(
         pos,
         n,
         vect_mpi_t,
         0,
         comm
   );


   MPI_Scatter(
         vel,
         loc_n,
         vect_mpi_t,

         loc_vel,
         loc_n,
         vect_mpi_t,

         0,
         comm
   );

}


/*==========================================================*/
/* OUTPUT SYSTEM STATE                                      */
/*==========================================================*/

void Output_state(
      double time,
      double masses[],
      vect_t pos[],
      vect_t loc_vel[],
      int n,
      int loc_n
) {

   int part;


   /*
    * Preserve the exact output design from the
    * supplied starter program.
    */
   MPI_Gather(
         loc_vel,
         loc_n,
         vect_mpi_t,

         vel,
         loc_n,
         vect_mpi_t,

         0,
         comm
   );


   if (my_rank == 0) {

      printf(
            "%.2f\n",
            time
      );


      for (part = 0;
           part < n;
           part++) {

         printf(
               "%3d %10.3e ",
               part,
               pos[part][X]
         );


         printf(
               "  %10.3e ",
               pos[part][Y]
         );


         printf(
               "  %10.3e ",
               vel[part][X]
         );


         printf(
               "  %10.3e\n",
               vel[part][Y]
         );
      }


      printf(
            "\n"
      );
   }


   /*
    * masses is intentionally retained in the
    * signature to remain compatible with the
    * supplied starter implementation.
    */
   (void)masses;

}


/*==========================================================*/
/* COMPUTE FORCE                                            */
/*==========================================================*/

void Compute_force(
      int loc_part,
      double masses[],
      vect_t loc_forces[],
      vect_t pos[],
      int n,
      int loc_n
) {

   int k;
   int part;

   double mg;

   vect_t f_part_k;

   double len;
   double len_3;
   double fact;


   /*
    * Convert local particle index to the
    * global particle index.
    */
   part =
         my_rank * loc_n +
         loc_part;


   loc_forces[loc_part][X] =
         0.0;

   loc_forces[loc_part][Y] =
         0.0;


#ifdef DEBUG

   printf(
         "Proc %d > Current total force on part %d "
         "= (%.3e, %.3e)\n",
         my_rank,
         part,
         loc_forces[loc_part][X],
         loc_forces[loc_part][Y]
   );

#endif


   /*
    * Preserve the original basic N-body algorithm.
    *
    * Every local particle interacts with every
    * other global particle.
    */
   for (k = 0;
        k < n;
        k++) {


      /*
       * Exclude self-interaction.
       */
      if (k != part) {


         f_part_k[X] =
               pos[part][X] -
               pos[k][X];


         f_part_k[Y] =
               pos[part][Y] -
               pos[k][Y];


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
               masses[part] *
               masses[k];


         fact =
               mg /
               len_3;


         f_part_k[X] *=
               fact;


         f_part_k[Y] *=
               fact;


#ifdef DEBUG

         printf(
               "Proc %d > Force on part %d due to part %d "
               "= (%.3e, %.3e)\n",
               my_rank,
               part,
               k,
               f_part_k[X],
               f_part_k[Y]
         );

#endif


         loc_forces[loc_part][X] +=
               f_part_k[X];


         loc_forces[loc_part][Y] +=
               f_part_k[Y];
      }
   }

}


/*==========================================================*/
/* UPDATE PARTICLE                                          */
/*==========================================================*/

void Update_part(
      int loc_part,
      double masses[],
      vect_t loc_forces[],
      vect_t loc_pos[],
      vect_t loc_vel[],
      int n,
      int loc_n,
      double delta_t
) {

   int part;

   double fact;


   part =
         my_rank * loc_n +
         loc_part;


   fact =
         delta_t /
         masses[part];


#ifdef DEBUG

   printf(
         "Proc %d > Before update of %d:\n",
         my_rank,
         part
   );


   printf(
         "   Position  = (%.3e, %.3e)\n",
         loc_pos[loc_part][X],
         loc_pos[loc_part][Y]
   );


   printf(
         "   Velocity  = (%.3e, %.3e)\n",
         loc_vel[loc_part][X],
         loc_vel[loc_part][Y]
   );


   printf(
         "   Net force = (%.3e, %.3e)\n",
         loc_forces[loc_part][X],
         loc_forces[loc_part][Y]
   );

#endif


   /*
    * IMPORTANT:
    *
    * Keep the same Euler update ordering as the
    * supplied mpi_nbody_basic.c:
    *
    * first update POSITION using the OLD velocity;
    * then update VELOCITY using the force.
    *
    * Changing this ordering changes the numerical
    * reference result.
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


#ifdef DEBUG

   printf(
         "Proc %d > Position of %d = "
         "(%.3e, %.3e), Velocity = (%.3e, %.3e)\n",
         my_rank,
         part,
         loc_pos[loc_part][X],
         loc_pos[loc_part][Y],
         loc_vel[loc_part][X],
         loc_vel[loc_part][Y]
   );

#endif


   /*
    * n is part of the original function interface.
    */
   (void)n;

}