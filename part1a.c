/*
 * part1a.c
 *
 * Part 1A - MPI Ring Communication
 *
 * Each MPI process owns one local block of bodies.
 * Position/mass data is circulated around the processes
 * using point-to-point ring communication.
 *
 * MPI_Sendrecv is used to avoid deadlock.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define G 6.673e-11
#define DT 1.0


typedef struct
{
    double mass;
    double x;
    double y;
    double vx;
    double vy;
} Body;


typedef struct
{
    double mass;
    double x;
    double y;
} PositionBlock;


/* ---------------------------------------------------------
   Calculate force contribution from one block
   --------------------------------------------------------- */

static void calculate_forces(
    const Body *local_bodies,
    int local_n,
    const PositionBlock *other,
    int other_n,
    int owner_rank,
    int my_rank,
    double *fx,
    double *fy
)
{
    int i;
    int j;

    for (i = 0; i < local_n; i++)
    {
        for (j = 0; j < other_n; j++)
        {
            /*
             * Do not allow a body to interact with itself.
             */
            if (owner_rank == my_rank && i == j)
            {
                continue;
            }

            double dx =
                other[j].x -
                local_bodies[i].x;

            double dy =
                other[j].y -
                local_bodies[i].y;

            double dist_sq =
                dx * dx +
                dy * dy;

            /*
             * Defensive check against division by zero.
             */
            if (dist_sq == 0.0)
            {
                continue;
            }

            double dist =
                sqrt(dist_sq);

            double dist_cubed =
                dist_sq * dist;

            /*
             * F = G m1 m2 / r^2
             *
             * x and y components:
             *
             * Fx = G m1 m2 dx / r^3
             * Fy = G m1 m2 dy / r^3
             */

            double factor =
                G *
                local_bodies[i].mass *
                other[j].mass /
                dist_cubed;

            fx[i] += factor * dx;
            fy[i] += factor * dy;
        }
    }
}


/* ---------------------------------------------------------
   Update local velocity and position
   --------------------------------------------------------- */

static void update_bodies(
    Body *local_bodies,
    int local_n,
    const double *fx,
    const double *fy
)
{
    int i;

    for (i = 0; i < local_n; i++)
    {
        double ax =
            fx[i] /
            local_bodies[i].mass;

        double ay =
            fy[i] /
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


/* ---------------------------------------------------------
   Print state snapshot

   IMPORTANT:
   Keep the output numeric and predictable for the grader.
   --------------------------------------------------------- */

static void print_state(
    const Body *bodies,
    int n,
    int step
)
{
    int i;

    printf("Step %d\n", step);

    for (i = 0; i < n; i++)
    {
        printf(
            "%.6e %.6e %.6e %.6e %.6e\n",
            bodies[i].mass,
            bodies[i].x,
            bodies[i].y,
            bodies[i].vx,
            bodies[i].vy
        );
    }

    printf("\n");
}


/* =========================================================
   Main
   ========================================================= */

int main(int argc, char *argv[])
{
    int rank;
    int comm_sz;

    int n;
    int steps;

    int local_n;

    int left;
    int right;

    int step;
    int round;
    int i;


    MPI_Init(&argc, &argv);

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank
    );

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &comm_sz
    );


    /*
     * The autograder local hints use:
     *
     * part1a n steps
     *
     * Example:
     *
     * mpirun -np 2 ./part1a 4 10
     */
    if (argc < 3)
    {
        if (rank == 0)
        {
            fprintf(
                stderr,
                "Usage: %s n steps\n",
                argv[0]
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }


    n =
        atoi(argv[1]);

    steps =
        atoi(argv[2]);


    if (n <= 0 ||
        steps < 0 ||
        n % comm_sz != 0)
    {
        if (rank == 0)
        {
            fprintf(
                stderr,
                "Invalid n, steps, or process count.\n"
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }


    local_n =
        n / comm_sz;


    /*
     * Ring neighbours.
     */
    left =
        (rank - 1 + comm_sz)
        % comm_sz;

    right =
        (rank + 1)
        % comm_sz;


    /* -----------------------------------------------------
       MPI datatypes
       ----------------------------------------------------- */

    MPI_Datatype MPI_BODY;
    MPI_Datatype MPI_POSITION;

    MPI_Type_contiguous(
        5,
        MPI_DOUBLE,
        &MPI_BODY
    );

    MPI_Type_commit(
        &MPI_BODY
    );


    MPI_Type_contiguous(
        3,
        MPI_DOUBLE,
        &MPI_POSITION
    );

    MPI_Type_commit(
        &MPI_POSITION
    );


    /* -----------------------------------------------------
       Rank 0 reads global input from stdin
       ----------------------------------------------------- */

    Body *all_bodies = NULL;


    if (rank == 0)
    {
        all_bodies =
            malloc(
                n * sizeof(Body)
            );

        if (all_bodies == NULL)
        {
            fprintf(
                stderr,
                "Memory allocation failed.\n"
            );

            MPI_Abort(
                MPI_COMM_WORLD,
                EXIT_FAILURE
            );
        }


        for (i = 0; i < n; i++)
        {
            if (
                scanf(
                    "%lf %lf %lf %lf %lf",
                    &all_bodies[i].mass,
                    &all_bodies[i].x,
                    &all_bodies[i].y,
                    &all_bodies[i].vx,
                    &all_bodies[i].vy
                ) != 5
            )
            {
                fprintf(
                    stderr,
                    "Failed to read body %d.\n",
                    i
                );

                MPI_Abort(
                    MPI_COMM_WORLD,
                    EXIT_FAILURE
                );
            }
        }
    }


    /* -----------------------------------------------------
       Each rank stores only its own local block for
       computation.
       ----------------------------------------------------- */

    Body *local_bodies =
        malloc(
            local_n * sizeof(Body)
        );

    PositionBlock *send_block =
        malloc(
            local_n *
            sizeof(PositionBlock)
        );

    PositionBlock *recv_block =
        malloc(
            local_n *
            sizeof(PositionBlock)
        );

    double *fx =
        calloc(
            local_n,
            sizeof(double)
        );

    double *fy =
        calloc(
            local_n,
            sizeof(double)
        );


    if (
        local_bodies == NULL ||
        send_block == NULL ||
        recv_block == NULL ||
        fx == NULL ||
        fy == NULL
    )
    {
        MPI_Abort(
            MPI_COMM_WORLD,
            EXIT_FAILURE
        );
    }


    /* -----------------------------------------------------
       Distribute initial bodies.
       ----------------------------------------------------- */

    MPI_Scatter(
        all_bodies,
        local_n,
        MPI_BODY,

        local_bodies,
        local_n,
        MPI_BODY,

        0,
        MPI_COMM_WORLD
    );


    /*
     * Print the initial state.
     *
     * This is useful because the grader refers to
     * "state snapshots".
     */
    if (rank == 0)
    {
        print_state(
            all_bodies,
            n,
            0
        );
    }


    /* =====================================================
       Main timestep loop
       ===================================================== */

    for (step = 1;
         step <= steps;
         step++)
    {
        /*
         * Reset accumulated forces.
         */
        for (i = 0;
             i < local_n;
             i++)
        {
            fx[i] = 0.0;
            fy[i] = 0.0;


            send_block[i].mass =
                local_bodies[i].mass;

            send_block[i].x =
                local_bodies[i].x;

            send_block[i].y =
                local_bodies[i].y;
        }


        /*
         * The first block belongs to this rank.
         */
        int owner =
            rank;


        /* -------------------------------------------------
           Ring traversal
           ------------------------------------------------- */

        for (round = 0;
             round < comm_sz;
             round++)
        {
            /*
             * Use the block currently held by this process.
             */
            calculate_forces(
                local_bodies,
                local_n,

                send_block,
                local_n,

                owner,
                rank,

                fx,
                fy
            );


            /*
             * Once all process blocks have been used,
             * there is no reason to send another block.
             */
            if (round ==
                comm_sz - 1)
            {
                break;
            }


            /*
             * Send one LOCAL POSITION BLOCK to the
             * right neighbour and receive one block
             * from the left neighbour.
             *
             * This is point-to-point ring communication
             * and avoids deadlock.
             */
            MPI_Sendrecv(
                send_block,
                local_n,
                MPI_POSITION,

                right,
                0,

                recv_block,
                local_n,
                MPI_POSITION,

                left,
                0,

                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );


            /*
             * Swap buffer pointers.
             *
             * No full-state collective communication is
             * used during the timestep loop.
             */
            PositionBlock *tmp =
                send_block;

            send_block =
                recv_block;

            recv_block =
                tmp;


            owner =
                (owner - 1 +
                 comm_sz)
                % comm_sz;
        }


        /* -------------------------------------------------
           Update local state after ALL force contributions
           have been accumulated.
           ------------------------------------------------- */

        update_bodies(
            local_bodies,
            local_n,
            fx,
            fy
        );


        /* -------------------------------------------------
           Gather only for producing the required output.
           It is not used to perform the ring computation.
           ------------------------------------------------- */

        MPI_Gather(
            local_bodies,
            local_n,
            MPI_BODY,

            all_bodies,
            local_n,
            MPI_BODY,

            0,
            MPI_COMM_WORLD
        );


        if (rank == 0)
        {
            print_state(
                all_bodies,
                n,
                step
            );
        }
    }


    /* -----------------------------------------------------
       Cleanup
       ----------------------------------------------------- */

    if (rank == 0)
    {
        free(
            all_bodies
        );
    }


    free(
        local_bodies
    );

    free(
        send_block
    );

    free(
        recv_block
    );

    free(
        fx
    );

    free(
        fy
    );


    MPI_Type_free(
        &MPI_BODY
    );

    MPI_Type_free(
        &MPI_POSITION
    );


    MPI_Finalize();

    return EXIT_SUCCESS;
}