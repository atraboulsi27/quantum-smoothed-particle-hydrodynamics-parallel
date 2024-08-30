#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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

void linspace(double start, double end, int num, double* v)
{
    double step = (end - start) / (num - 1);

    #pragma omp parallel for
    for (int i = 0; i < num; i++)
    {
        v[i] = start + i * step;
    }
}

void density(double* v, double* rho, size_t n, double m, double h)
{
    double uij, rho_ij;

    #pragma omp parallel for private(uij, rho_ij)
    for (int i = 0; i < n; i++)
    {
        rho[i] = 0.0;

        for (int j = 0; j < n; j++)
        {
            uij = v[i] - v[j];
            rho_ij = m * kernel(uij, h, 0);
            rho[i] += rho_ij;
        }
    }
}

double pressure(double* x, double* rho, double *P, size_t n, double m, double h)
{
    double* drho = malloc(n * sizeof(double));
    double* ddrho = malloc(n * sizeof(double));

    double drho_ij, ddrho_ij, P_ij, uij;

    #pragma omp parallel for private(uij, drho_ij, ddrho_ij)
    for (int i = 0; i < n; i++)
    {
        drho[i] = 0;
        ddrho[i] = 0;

        for (int j = 0; j < n; j++)
        {
            uij = x[i] - x[j];
            drho_ij = m * kernel(uij, h, 1);
            ddrho_ij = m * kernel(uij, h, 2);

            drho[i] += drho_ij;
            ddrho[i] += ddrho_ij;
        }
    }

    #pragma omp parallel for private(uij, P_ij)
    for (int i = 0; i < n; i++)
    {
        P[i] = 0;

        for (int j = 0; j < n; j++)
        {
            uij = x[i] - x[j];
            P_ij = 0.25 * (pow(drho[j], 2) / rho[j] - ddrho[j]) * m / rho[j] * kernel(uij, h, 0);
            P[i] += P_ij;
        }
    }

    free(drho);
    free(ddrho);
}

double acceleration(double* x, double* u, double* a, double m, double* rho, double* P, double b, double h, size_t n)
{
    double uij, fac, pressure_a;

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        a[i] = 0;

    #pragma omp parallel for private(uij, fac, pressure_a)
    for (int i = 0; i < n; i++)
    {
        a[i] = a[i] - u[i] * b - x[i];

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
            a[i] = a[i] + pressure_a;
        }

        free(x_js);
        free(P_js);
        free(rho_js);
    }
}

double probeDensity(double* x, double m, double h, size_t n, double* xx, double* rr)
{
    double uij, rho_ij;

    #pragma omp parallel for private(uij, rho_ij)
    for (int i = 0; i < 400; i++)
    {
        rr[i] = 0.0;

        for (int j = 0; j < n; j++)
        {
            uij = xx[i] - x[j];
            rho_ij = m * kernel(uij, h, 0);
            rr[i] += rho_ij;
        }
    }
}

double probeDensityExact(double* rr, double t, double* xx)
{
    #pragma omp parallel for
    for (int i = 0; i < 400; i++)
    {
        rr[i] = 1.0 / sqrt(M_PI) * exp(-pow(xx[i] - sin(t), 2) / 2.0) * exp(-pow(xx[i] - sin(t), 2) / 2.0);
    }
}

int main()
{
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
    file = fopen("results.txt", "w");

    double* x = malloc(n * sizeof(double));
    linspace(-3.0, 3.0, n, x); // this is used to compute x

    double* u = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        u[i] = 0.0;
    }

    double* rho = malloc(n * sizeof(double));
    density(x, rho, n, m, h); // this is used to compute rho

    double* P = malloc(n * sizeof(double));
    pressure(x, rho, P, n, m, h); // this is used to compute P

    double* a = malloc(n * sizeof(double)); 
    acceleration(x, u, a, m, rho, P, b, h, n); // this is used to compute accelerations

    double* u_mhalf = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        u_mhalf[i] = u[i] - 0.5 * dt * a[i];
    }

    double* xx = malloc(nt_setup * sizeof(double));
    linspace(-4.0, 4.0, nt_setup, xx);

    double* u_phalf = malloc(n * sizeof(double));
    double* rr = malloc(400 * sizeof(double));
    double* rr_exact = malloc(400 * sizeof(double));

    for (int i = (-1) * nt_setup; i < nt; i++)
    {
        #pragma omp parallel for
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
            #pragma omp parallel for
            for (int j = 0; j < n; j++)
            {
                u[j] = 1.0;
                u_mhalf[j] = u[j];
                b = 0;
            }
        }

        density(x, rho, n, m, h); // this is used to compute rho
        pressure(x, rho, P, n, m, h); // this is used to compute P
        acceleration(x, u, a, m, rho, P, b, h, n); // this is used to compute accelerations

        if ((i >= 0) && (i % n_out == 0))
        {
            probeDensity(x, m, h, n, xx, rr);
            probeDensityExact(rr_exact, t, xx);

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

    fclose(file);

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

    return 0;
}
