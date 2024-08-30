#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define M_PI 3.14159265358979323846

double kernel(double r, double h, int deriv)
{
    const double sqrt_pi_inv = 1.0 / sqrt(M_PI);
    double exp_part = exp((-1) * r * r / (h * h));

    switch (deriv) {
        case 0:
            return pow(h, -1) / sqrt(M_PI) * exp_part;
        case 1:
            return pow(h, -3) / sqrt(M_PI) * exp_part * (-2 * r);
        case 2:
            return pow(h, -5) / sqrt(M_PI) * exp_part * (4 * pow(r,2) - 2 * pow(h,2));
        case 3:
            return pow(h, -7) / sqrt(M_PI) * exp_part * (-8 * pow(r, 3) + 12 * pow(h, 2) * r);
    }
    return 0.0; // default return value if deriv is not 0, 1, 2, or 3
}

void linspace(double start, double end, int num, double* v, int rank, int size)
{
    double step = (end - start) / (num - 1);
    int iter_per_process = num / size; // divide the data according to the number of processes
    int remainder = num % size; // compute the remainder

    int local_start = rank * iter_per_process + (rank < remainder ? rank : remainder); //compute the start of each process
    int local_end = local_start + iter_per_process + (rank < remainder ? 1 : 0); // compute the end of each process

    double* local_v = malloc((local_end - local_start) * sizeof(double)); // allocate local vector

    for (int local_iter = local_start; local_iter < local_end; local_iter++)
    {
        local_v[local_iter - local_start] = start + local_iter * step; //compute local iterations
    }

    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }

    int local_count = local_end - local_start;
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD); // since we are dealing with non perfectly divisable arrayss
                                                                                     // send how many elements each process processed

    if (rank == 0)
    {
        displs[0] = 0;
        for (int i = 1; i < size; i++)
        {
            displs[i] = displs[i - 1] + recvcounts[i - 1]; // create a checkpoint array of the number of elements were processed at different ranks
        }
    }

    MPI_Gatherv(local_v, local_count, MPI_DOUBLE, v, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); // merge the locally processed vectors into one main vector

    if (rank == 0)
    {
        free(recvcounts);
        free(displs);
    }
    free(local_v);
}

void density(double* v, double* rho, size_t n, double m, double h, int rank, int size)
{
    int iter_per_process = n / size;
    int remainder = n % size;

    int local_start = rank * iter_per_process + (rank < remainder ? rank : remainder);
    int local_end = local_start + iter_per_process + (rank < remainder ? 1 : 0);

    double* local_rho = malloc((local_end - local_start) * sizeof(double));

    for (int local_iter = local_start; local_iter < local_end; local_iter++)
    {
        local_rho[local_iter - local_start] = 0.0;

        for (int j = 0; j < n; j++)
        {
            double uij = v[local_iter] - v[j];
            double rho_ij = m * kernel(uij, h, 0);
            local_rho[local_iter - local_start] += rho_ij;
        }
    }

    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }

    int local_count = local_end - local_start;
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        displs[0] = 0;
        for (int i = 1; i < size; i++)
        {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    MPI_Gatherv(local_rho, local_count, MPI_DOUBLE, rho, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        free(recvcounts);
        free(displs);
    }
    free(local_rho);
}


void pressure(double* x, double* rho, double* P, size_t n, double m, double h, int rank, int size)
{
    int iter_per_process = n / size;
    int remainder = n % size;

    int local_start = rank * iter_per_process + (rank < remainder ? rank : remainder);
    int local_end = local_start + iter_per_process + (rank < remainder ? 1 : 0);

    double* local_drho = malloc((local_end - local_start) * sizeof(double));
    double* local_ddrho = malloc((local_end - local_start) * sizeof(double));
    double* drho = malloc(n * sizeof(double));
    double* ddrho = malloc(n * sizeof(double));
    
    double drho_ij, ddrho_ij, P_ij, uij;

    for (int i = local_start; i < local_end; i++)
    {
        local_drho[i - local_start] = 0;
        local_ddrho[i - local_start] = 0;

        for (int j = 0; j < n; j++)
        {
            uij = x[i] - x[j];
            drho_ij = m * kernel(uij, h, 1);
            ddrho_ij = m * kernel(uij, h, 2);

            local_drho[i - local_start] += drho_ij;
            local_ddrho[i - local_start] += ddrho_ij;
        }
    }

    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        for (int i = 0; i < size; i++)
        {
            recvcounts[i] = iter_per_process + (i < remainder ? 1 : 0);
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
        }
    }

    MPI_Gatherv(local_drho, local_end - local_start, MPI_DOUBLE, drho, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_ddrho, local_end - local_start, MPI_DOUBLE, ddrho, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(drho, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ddrho, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);


    free(local_drho);
    free(local_ddrho);

    double* local_P = malloc((local_end - local_start) * sizeof(double));

    for (int i = local_start; i < local_end; i++)
    {
        local_P[i - local_start] = 0;
        for (int j = 0; j < n; j++)
        {
            uij = x[i] - x[j];
            P_ij = 0.25 * (pow(drho[j], 2) / rho[j] - ddrho[j]) * m / rho[j] * kernel(uij, h, 0);
            local_P[i - local_start] += P_ij;
        }
    }

    MPI_Gatherv(local_P, local_end - local_start, MPI_DOUBLE, P, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        free(recvcounts);
        free(displs);
    }
    free(local_P);
    free(drho);
    free(ddrho);
}

void acceleration(double* x, double* u, double* a, double m, double* rho, double* P, double b, double h, size_t n, int rank, int size)
{
    double uij, fac, pressure_a;

    int iter_per_process = n / size;
    int remainder = n % size;

    int local_start = rank * iter_per_process + (rank < remainder ? rank : remainder);
    int local_end = local_start + iter_per_process + (rank < remainder ? 1 : 0);

    double* local_a = malloc((local_end - local_start) * sizeof(double));


    for (int i = local_start; i < local_end; i++)
        local_a[i-local_start] = 0;

    for (int i = local_start; i < local_end; i++)
    {
        local_a[i - local_start] = local_a[i - local_start] - u[i] * b - x[i];

        double* x_js = (double*)malloc((n - 1) * sizeof(double));
        double* P_js = (double*)malloc((n - 1) * sizeof(double));
        double* rho_js = (double*)malloc((n - 1) * sizeof(double));

        // Populate arrays x_js, P_js, rho_js without element i
        int idx = 0;
        for (int j = 0; j < n; j++) {
            if (j != i) {
                x_js[idx] = x[j];
                P_js[idx] = P[j];
                rho_js[idx] = rho[j];
                idx++;
            }
        }

        // Calculate acceleration due to pressure
        for (int j = 0; j < n - 1; j++) {
            uij = x[i] - x_js[j];
            fac = -m * (P[i] / (rho[i] * rho[i]) + P_js[j] / (rho_js[j] * rho_js[j]));
            pressure_a = fac * kernel(uij, h, 1);
            local_a[i - local_start] = local_a[i - local_start] + pressure_a;
        }

        free(x_js);
        free(P_js);
        free(rho_js);
    }

    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        for (int i = 0; i < size; i++)
        {
            recvcounts[i] = iter_per_process + (i < remainder ? 1 : 0);
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
        }
    }

    MPI_Gatherv(local_a, local_end - local_start, MPI_DOUBLE, a, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        free(recvcounts);
        free(displs);
    }
    free(local_a);
}

void probeDensity(double* x, double m, double h, size_t n, double* xx, double* rr, int rank, int size)
{
    double uij, rho_ij;

    int iter_per_process = 400 / size;
    int remainder = 400 % size;

    int local_start = rank * iter_per_process + (rank < remainder ? rank : remainder);
    int local_end = local_start + iter_per_process + (rank < remainder ? 1 : 0);

    double* local_rr = malloc((local_end - local_start) * sizeof(double));

    for (int i = local_start; i < local_end; i++)
    {
        local_rr[i - local_start] = 0.0;

        for (int j = 0; j < n; j++)
        {
            uij = xx[i] - x[j];
            rho_ij = m * kernel(uij, h, 0);
            local_rr[i - local_start] += rho_ij;
        }
    }
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        for (int i = 0; i < size; i++)
        {
            recvcounts[i] = iter_per_process + (i < remainder ? 1 : 0);
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
        }
    }

    MPI_Gatherv(local_rr, local_end - local_start, MPI_DOUBLE, rr, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        free(recvcounts);
        free(displs);
    }
    free(local_rr);
}

void probeDensityExact(double* rr, double t, double* xx, int rank, int size)
{

    double uij, rho_ij;

    int iter_per_process = 400 / size;
    int remainder = 400 % size;

    int local_start = rank * iter_per_process + (rank < remainder ? rank : remainder);
    int local_end = local_start + iter_per_process + (rank < remainder ? 1 : 0);

    double* local_rr = malloc((local_end - local_start) * sizeof(double));

    for (int i = local_start; i < local_end; i++)
    {
        local_rr[i-local_start] = 1.0 / sqrt(M_PI) * exp(-pow(xx[i] - sin(t), 2) / 2.0) * exp(-pow(xx[i] - sin(t), 2) / 2.0);
    }

    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        for (int i = 0; i < size; i++)
        {
            recvcounts[i] = iter_per_process + (i < remainder ? 1 : 0);
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
        }
    }

    MPI_Gatherv(local_rr, local_end - local_start, MPI_DOUBLE, rr, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        free(recvcounts);
        free(displs);
    }
    free(local_rr);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 100;
    double dt = 0.02;
    int nt = 10000;
    int nt_setup = 400;
    int n_out = 25;
    double b = 4;
    double m = (double) 1.0 / n;
    double h = (double) 40 / n;
    double t = 0.0;

    double rr_list[4][400];
    double rr_exact_list[4][400];

    FILE* file;
    if (rank == 0)
    {
        file = fopen("results.txt", "w");
    }

    double* x = malloc(n * sizeof(double));
    linspace(-3.0, 3.0, n, x, rank, size);

    double* u = malloc(n * sizeof(double));
    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            u[i] = 0.0;
        }
    }
    MPI_Bcast(u, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* rho = malloc(n * sizeof(double));
    density(x, rho, n, m, h, rank, size); // This will need to be parallelized with MPI

    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(rho, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double* P = malloc(n * sizeof(double));
    pressure(x, rho, P, n, m, h, rank, size); // This will need to be parallelized with MPI

    MPI_Bcast(P, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double* a = malloc(n * sizeof(double)); 
    acceleration(x, u, a, m, rho, P, b, h, n, rank, size); // This will need to be parallelized with MPI
    MPI_Bcast(a, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    double* u_mhalf = malloc(n * sizeof(double));
    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            u_mhalf[i] = u[i] - 0.5 * dt * a[i];
        }
    }
    MPI_Bcast(u_mhalf, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* xx = malloc(nt_setup * sizeof(double));
    linspace(-4.0, 4.0, nt_setup, xx, rank, size);
    MPI_Bcast(xx, nt_setup, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* u_phalf = malloc(n * sizeof(double));
    double* rr = malloc(400 * sizeof(double));
    double* rr_exact = malloc(400 * sizeof(double));

    for (int i = (-1) * nt_setup; i < nt; i++)
    {
        for (int j = 0; j < n; j++)
        {
            u_phalf[j] = u_mhalf[j] + a[j] * dt;
            x[j] = x[j] + u_phalf[j] * dt;
            u[j] = 0.5 * (u_mhalf[j] + u_phalf[j]);
            u_mhalf[j] = u_phalf[j];
        }

        if (i >= 0)
            t += dt;

        if (i == -1)
        {
            for (int j = 0; j < n; j++)
            {
                u[j] = 1.0;
                u_mhalf[j] = u[j];
                b = 0;
            }
        }

        MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        density(x, rho, n, m, h, rank, size); // This will need to be parallelized with MPI
        MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(rho, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        pressure(x, rho, P, n, m, h, rank, size); // This will need to be parallelized with MPI
        MPI_Bcast(P, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        acceleration(x, u, a, m, rho, P, b, h, n, rank, size); // This will need to be parallelized with MPI
        MPI_Bcast(a, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        probeDensity(x, m, h, n, xx, rr, rank, size);
        probeDensityExact(rr_exact, t, xx, rank, size);


        if ((i >= 0) && (i % n_out == 0) && (rank == 0))
        {

            for (int j = 0; j < 400; j++)
            {
                fprintf(file, "%f ", rr[j]);
            }
            fprintf(file, "\n");
            for (int j = 0; j < 400; j++)
            {
                fprintf(file, "%f ", rr_exact[j]);
            }
            fprintf(file, "\n");
        }
    }

    if (rank == 0)
    {
        fclose(file);
    }

    free(x);
    free(u);
    free(rho);
    free(P);
    free(a);
    free(u_mhalf);
    free(xx);
    free(u_phalf);
    free(rr);
    free(rr_exact);

    MPI_Finalize();
    return 0;
}
