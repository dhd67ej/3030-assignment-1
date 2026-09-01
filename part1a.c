/*
 * part1a.c
 *
 * Assignment Part 1A - MPI Ring Communication
 *
 * This program implements a distributed N-body simulation using
 * point-to-point MPI communication arranged in a ring topology.
 *
 * Each MPI process:
 *   1. Owns a local block of bodies.
 *   2. Computes forces from its own local block.
 *   3. Passes position/mass blocks around the MPI ring.
 *   4. Receives blocks owned by other processes.
 *   5. Accumulates their force contributions.
 *   6. Updates only its own local bodies.
 *
 * MPI_Sendrecv is used so the communication is deadlock-free.
 *
 * Compile:
 *      mpicc part1a.c -o part1a -lm
 *
 * Run with 4 MPI processes:
 *      mpirun -np 4 ./part1a
 *
 * Optional:
 *      mpirun -np 4 ./part1a 16 10
 *
 *      16 = number of bodies
 *      10 = number of simulation steps
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ---------------------------------------------------------
   Simulation constants
   --------------------------------------------------------- */

#define DEFAULT_N       8
#define DEFAULT_STEPS   5

#define DT              0.01
#define G               1.0
#define SOFTENING       1.0e-9

/*
 * Each body contains:
 * mass
 * position (x, y)
 * velocity (vx, vy)
 */
typedef struct
{
    double mass;

    double x;
    double y;

    double vx;
    double vy;

} Body;


/*
 * Only the data required for calculating gravitational
 * forces needs to travel around the ring.
 */
typedef struct
{
    double mass;
    double x;
    double y;

} BodyPosition;


/* ---------------------------------------------------------
   Function declarations
   --------------------------------------------------------- */

void initialise_bodies(Body *bodies, int n);

void copy_position_block(
    const Body *local_bodies,
    BodyPosition *block,
    int local_n
);

void compute_force_from_block(
    const Body *local_bodies,
    int local_n,
    const BodyPosition *remote_block,
    int block_n,
    int remote_owner,
    int my_rank,
    double *force_x,
    double *force_y
);

void update_local_bodies(
    Body *local_bodies,
    int local_n,
    double *force_x,
    double *force_y
);

void print_global_state(
    Body *local_bodies,
    int local_n,
    int n,
    int rank,
    int comm_sz,
    int step,
    MPI_Datatype mpi_body
);


/* =========================================================
   main
   ========================================================= */

int main(int argc, char *argv[])
{
    int rank;
    int comm_sz;

    int n = DEFAULT_N;
    int steps = DEFAULT_STEPS;

    int local_n;

    int left;
    int right;

    int round;
    int step;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);


    /* -----------------------------------------------------
       Read optional command-line arguments
       ----------------------------------------------------- */

    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        steps = atoi(argv[2]);
    }


    /* -----------------------------------------------------
       Basic input validation
       ----------------------------------------------------- */

    if (n <= 0 || steps < 0)
    {
        if (rank == 0)
        {
            fprintf(
                stderr,
                "Error: number of bodies must be positive "
                "and number of steps must not be negative.\n"
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }


    /*
     * This Part 1A implementation uses equal block
     * partitioning, so n must be divisible by the
     * number of MPI processes.
     */
    if (n % comm_sz != 0)
    {
        if (rank == 0)
        {
            fprintf(
                stderr,
                "Error: n (%d) must be evenly divisible "
                "by the number of MPI processes (%d).\n",
                n,
                comm_sz
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }


    local_n = n / comm_sz;


    /* -----------------------------------------------------
       Ring neighbours

       Example with 4 processes:

             P0 -> P1 -> P2 -> P3
              ^                 |
              |_________________|

       right:
           (rank + 1) % p

       left:
           (rank - 1 + p) % p
       ----------------------------------------------------- */

    right = (rank + 1) % comm_sz;

    left = (rank - 1 + comm_sz) % comm_sz;


    /* -----------------------------------------------------
       Create MPI datatype for BodyPosition

       BodyPosition consists of three consecutive doubles:
       mass, x and y.
       ----------------------------------------------------- */

    MPI_Datatype mpi_body_position;

    MPI_Type_contiguous(
        3,
        MPI_DOUBLE,
        &mpi_body_position
    );

    MPI_Type_commit(&mpi_body_position);


    /*
     * Body consists of five consecutive doubles.
     * This datatype is used only when gathering the final
     * state for printing.
     */
    MPI_Datatype mpi_body;

    MPI_Type_contiguous(
        5,
        MPI_DOUBLE,
        &mpi_body
    );

    MPI_Type_commit(&mpi_body);


    /* -----------------------------------------------------
       Allocate initial global array.

       For Part 1A we initialise the complete initial state
       and distribute equal blocks to the MPI processes.
       ----------------------------------------------------- */

    Body *all_bodies = NULL;

    if (rank == 0)
    {
        all_bodies = malloc(
            n * sizeof(Body)
        );

        if (all_bodies == NULL)
        {
            fprintf(
                stderr,
                "Unable to allocate initial body array.\n"
            );

            MPI_Abort(
                MPI_COMM_WORLD,
                EXIT_FAILURE
            );
        }

        initialise_bodies(
            all_bodies,
            n
        );
    }


    /* -----------------------------------------------------
       Allocate local data
       ----------------------------------------------------- */

    Body *local_bodies =
        malloc(local_n * sizeof(Body));

    BodyPosition *send_block =
        malloc(local_n * sizeof(BodyPosition));

    BodyPosition *recv_block =
        malloc(local_n * sizeof(BodyPosition));

    double *force_x =
        malloc(local_n * sizeof(double));

    double *force_y =
        malloc(local_n * sizeof(double));


    if (
        local_bodies == NULL ||
        send_block == NULL ||
        recv_block == NULL ||
        force_x == NULL ||
        force_y == NULL
    )
    {
        fprintf(
            stderr,
            "Process %d: memory allocation failed.\n",
            rank
        );

        MPI_Abort(
            MPI_COMM_WORLD,
            EXIT_FAILURE
        );
    }


    /* -----------------------------------------------------
       Distribute the initial bodies.

       Process 0 owns:
           bodies 0 ... local_n-1

       Process 1 owns:
           bodies local_n ... 2*local_n-1

       etc.
       ----------------------------------------------------- */

    MPI_Scatter(
        all_bodies,
        local_n,
        mpi_body,

        local_bodies,
        local_n,
        mpi_body,

        0,
        MPI_COMM_WORLD
    );


    if (rank == 0)
    {
        free(all_bodies);
        all_bodies = NULL;
    }


    /* -----------------------------------------------------
       Print initial distribution
       ----------------------------------------------------- */

    if (rank == 0)
    {
        printf(
            "\nMPI N-body Ring Communication\n"
        );

        printf(
            "Processes : %d\n",
            comm_sz
        );

        printf(
            "Bodies    : %d\n",
            n
        );

        printf(
            "Local n   : %d\n",
            local_n
        );

        printf(
            "Steps     : %d\n\n",
            steps
        );
    }


    MPI_Barrier(MPI_COMM_WORLD);


    /* =====================================================
       MAIN SIMULATION LOOP
       ===================================================== */

    for (step = 0; step < steps; step++)
    {
        /*
         * Every process starts a time step with zero force
         * on each of its local bodies.
         */

        for (int i = 0; i < local_n; i++)
        {
            force_x[i] = 0.0;
            force_y[i] = 0.0;
        }


        /*
         * Copy this process's own positions into the
         * communication block.
         */
        copy_position_block(
            local_bodies,
            send_block,
            local_n
        );


        /*
         * At the beginning, the data block is owned by
         * this MPI process.
         */
        int block_owner = rank;


        /* =================================================
           RING ALGORITHM

           Each process needs contributions from all p
           blocks.

           Round 0:
               use own block

           Then circulate blocks:

               send -> right neighbour
               receive <- left neighbour

           MPI_Sendrecv performs both operations safely.
           ================================================= */

        for (round = 0; round < comm_sz; round++)
        {
            /* ---------------------------------------------
               Calculate force contributions from the
               block currently stored in send_block.
               --------------------------------------------- */

            compute_force_from_block(
                local_bodies,
                local_n,
                send_block,
                local_n,
                block_owner,
                rank,
                force_x,
                force_y
            );


            /*
             * No communication is required after the
             * final block has been processed.
             */
            if (round == comm_sz - 1)
            {
                break;
            }


            /* ---------------------------------------------
               Send current block to RIGHT.

               Receive another block from LEFT.

               MPI_Sendrecv avoids deadlock because the
               send and receive are matched within one MPI
               operation.
               --------------------------------------------- */

            MPI_Sendrecv(
                send_block,
                local_n,
                mpi_body_position,
                right,
                0,

                recv_block,
                local_n,
                mpi_body_position,
                left,
                0,

                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );


            /*
             * The received block came from the process
             * immediately to the left of the current block
             * owner in the ring.
             */

            block_owner =
                (block_owner - 1 + comm_sz)
                % comm_sz;


            /*
             * Swap send and receive buffers instead of
             * copying the entire block.
             */
            BodyPosition *temp;

            temp = send_block;
            send_block = recv_block;
            recv_block = temp;
        }


        /* -------------------------------------------------
           All force contributions have now been accumulated.

           Update only the bodies owned by this process.
           ------------------------------------------------- */

        update_local_bodies(
            local_bodies,
            local_n,
            force_x,
            force_y
        );


        /*
         * This synchronisation ensures every process has
         * completed the current time step before the next
         * time step begins.
         */
        MPI_Barrier(
            MPI_COMM_WORLD
        );
    }


    /* -----------------------------------------------------
       Gather and print final result.
       ----------------------------------------------------- */

    print_global_state(
        local_bodies,
        local_n,
        n,
        rank,
        comm_sz,
        steps,
        mpi_body
    );


    /* -----------------------------------------------------
       Clean up
       ----------------------------------------------------- */

    free(local_bodies);
    free(send_block);
    free(recv_block);
    free(force_x);
    free(force_y);

    MPI_Type_free(
        &mpi_body_position
    );

    MPI_Type_free(
        &mpi_body
    );

    MPI_Finalize();

    return EXIT_SUCCESS;
}


/* =========================================================
   Initialise bodies

   Deterministic values are deliberately used so that
   different runs give the same result.
   ========================================================= */

void initialise_bodies(
    Body *bodies,
    int n
)
{
    for (int i = 0; i < n; i++)
    {
        bodies[i].mass =
            1.0 + 0.1 * (double)i;

        bodies[i].x =
            (double)i;

        bodies[i].y =
            0.5 * (double)(i % 3);

        bodies[i].vx =
            0.0;

        bodies[i].vy =
            0.0;
    }
}


/* =========================================================
   Copy local bodies into communication block
   ========================================================= */

void copy_position_block(
    const Body *local_bodies,
    BodyPosition *block,
    int local_n
)
{
    for (int i = 0; i < local_n; i++)
    {
        block[i].mass =
            local_bodies[i].mass;

        block[i].x =
            local_bodies[i].x;

        block[i].y =
            local_bodies[i].y;
    }
}


/* =========================================================
   Calculate gravitational force produced by one position
   block on this process's local bodies.
   ========================================================= */

void compute_force_from_block(
    const Body *local_bodies,
    int local_n,
    const BodyPosition *remote_block,
    int block_n,
    int remote_owner,
    int my_rank,
    double *force_x,
    double *force_y
)
{
    for (int i = 0; i < local_n; i++)
    {
        for (int j = 0; j < block_n; j++)
        {
            /*
             * If the block belongs to this process,
             * i == j represents the same physical body.
             * A body must not exert gravitational force
             * on itself.
             */
            if (
                remote_owner == my_rank &&
                i == j
            )
            {
                continue;
            }


            double dx =
                remote_block[j].x
                - local_bodies[i].x;

            double dy =
                remote_block[j].y
                - local_bodies[i].y;


            /*
             * Squared separation.
             * SOFTENING prevents division by zero or
             * extremely large numerical forces.
             */
            double distance_squared =
                dx * dx +
                dy * dy +
                SOFTENING;


            double distance =
                sqrt(distance_squared);


            /*
             * Newtonian gravitational force magnitude:
             *
             * F = G * m1 * m2 / r^2
             *
             * Direction is obtained using dx/r and dy/r.
             *
             * Therefore:
             *
             * Fx = G*m1*m2*dx/r^3
             * Fy = G*m1*m2*dy/r^3
             */

            double distance_cubed =
                distance_squared *
                distance;


            double scale =
                G *
                local_bodies[i].mass *
                remote_block[j].mass /
                distance_cubed;


            force_x[i] +=
                scale * dx;

            force_y[i] +=
                scale * dy;
        }
    }
}


/* =========================================================
   Update velocity and position of local bodies
   ========================================================= */

void update_local_bodies(
    Body *local_bodies,
    int local_n,
    double *force_x,
    double *force_y
)
{
    for (int i = 0; i < local_n; i++)
    {
        /*
         * Newton's second law:
         *
         * a = F / m
         */

        double ax =
            force_x[i] /
            local_bodies[i].mass;

        double ay =
            force_y[i] /
            local_bodies[i].mass;


        /*
         * Update velocity.
         */

        local_bodies[i].vx +=
            ax * DT;

        local_bodies[i].vy +=
            ay * DT;


        /*
         * Update position.
         */

        local_bodies[i].x +=
            local_bodies[i].vx * DT;

        local_bodies[i].y +=
            local_bodies[i].vy * DT;
    }
}


/* =========================================================
   Gather all final local blocks on process 0
   and display the global result.
   ========================================================= */

void print_global_state(
    Body *local_bodies,
    int local_n,
    int n,
    int rank,
    int comm_sz,
    int step,
    MPI_Datatype mpi_body
)
{
    Body *global_bodies = NULL;


    if (rank == 0)
    {
        global_bodies =
            malloc(n * sizeof(Body));

        if (global_bodies == NULL)
        {
            fprintf(
                stderr,
                "Unable to allocate final body array.\n"
            );

            MPI_Abort(
                MPI_COMM_WORLD,
                EXIT_FAILURE
            );
        }
    }


    MPI_Gather(
        local_bodies,
        local_n,
        mpi_body,

        global_bodies,
        local_n,
        mpi_body,

        0,
        MPI_COMM_WORLD
    );


    if (rank == 0)
    {
        printf(
            "Final state after %d step(s):\n\n",
            step
        );

        printf(
            "%-6s %-10s %-14s %-14s %-14s %-14s\n",
            "Body",
            "Mass",
            "X",
            "Y",
            "VX",
            "VY"
        );


        for (int i = 0; i < n; i++)
        {
            printf(
                "%-6d %-10.4f "
                "%-14.8f %-14.8f "
                "%-14.8f %-14.8f\n",

                i,

                global_bodies[i].mass,

                global_bodies[i].x,
                global_bodies[i].y,

                global_bodies[i].vx,
                global_bodies[i].vy
            );
        }


        /*
         * A checksum is useful when comparing executions
         * with different numbers of MPI processes.
         */
        double checksum = 0.0;

        for (int i = 0; i < n; i++)
        {
            checksum +=
                global_bodies[i].x +
                global_bodies[i].y +
                global_bodies[i].vx +
                global_bodies[i].vy;
        }


        printf(
            "\nChecksum: %.12f\n",
            checksum
        );


        free(global_bodies);
    }


    (void)comm_sz;
}
