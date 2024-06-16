#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>


__host__
void generate_tridiagonal_system(int n, float a[], float b[], float c[], float d[]) {
    // Seed the random number generator for variability in results
    // srand(time(NULL));

    for (int i = 0; i < n; i++) {
        // Generate random values for a, b, c, and d
        a[i] = (i > 0) ? rand() % 100 + 1 : 0;  // Upper diagonal (no entry at i=0)
        c[i] = (i < n-1) ? rand() % 100 + 1 : 0;  // Lower diagonal (no entry at i=n-1)
        b[i] = a[i] + c[i] + rand() % 100 + 50;  // Ensure diagonal dominance
        d[i] = rand() % 100 + 1;
    }
}

__global__ void list_print(int nmax, float * in) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    printf("Thread %i shows %f \n", i, in[i]);
}


__global__ void Solve_Kernel(
    float * alist, float * blist, float * clist, float * dlist, float * xlist,
    int iter_max, int DMax) {

    int idx_row = blockIdx.x*blockDim.x + threadIdx.x;
    int row_max = DMax - 1;

    int stride = 1;
    int next_stride = stride;

    float a1, b1, c1, d1;
    float k01, k21, c01, a21, d01, d21;

    bool next_or_ot = true;
    int accum;

    for (int iter = 0; iter < iter_max; iter++) {

        if ( next_or_ot ) {

            next_stride = stride<<1;

            // 1    for updating 'a'
            if ((idx_row - stride)<0) {
            // 1.1  if it is the 'first' line
                a1 = 0.0f;
                k01 = 0.0f;
                c01 = 0.0f;
                d01 = 0.0f;
            } else if ((idx_row - next_stride)<0) {
            // 1.2  if no place for 'a'
                a1 = 0.0f;
                k01 = alist[idx_row]/blist[idx_row - stride];
                c01 = clist[idx_row - stride]*k01;
                d01 = dlist[idx_row - stride]*k01;
            } else {
            // 1.3  for rest general rows
                k01 = alist[idx_row]/blist[idx_row - stride];
                a1 = -alist[idx_row - stride]*k01;
                c01 = clist[idx_row - stride]*k01;
                d01 = dlist[idx_row - stride]*k01;
            }

            // 2    for updating 'c'
            if ((idx_row + stride)>row_max) {
            // 2.1  if it is the 'last' line
                c1 = 0.0f;
                k21 = 0.0f;
                a21 = 0.0f;
                d21 = 0.0f;
            } else if ((idx_row + next_stride)>row_max) {
                c1 = 0.0f;
                k21 = clist[idx_row]/blist[idx_row + stride];
                a21 = alist[idx_row + stride]*k21;
                d21 = dlist[idx_row + stride]*k21;
            } else {
                k21 = clist[idx_row]/blist[idx_row + stride];
                c1 = -clist[idx_row + stride]*k21;
                a21 = alist[idx_row + stride]*k21;
                d21 = dlist[idx_row + stride]*k21;
            }
            // 3   for updating 'b'
            b1 = blist[idx_row] - c01 - a21;
            // 4   for updating 'd'
            d1 = dlist[idx_row] - d01 - d21;

            stride = next_stride;

            //Determine if this line has reached the bi-set
            int pos = idx_row-2*stride;
            accum = 0;
            for ( size_t iter = 0; iter<5; iter++ ) {
                if (pos >=0 && pos < DMax) accum++;
                pos+=stride;
            }
            if (accum < 3) {
                next_or_ot = false;//Turn of for ever
            }

        }

        __syncthreads();

        alist[idx_row] = a1;
        blist[idx_row] = b1;
        clist[idx_row] = c1;
        dlist[idx_row] = d1;

    }

    if ( accum==1 ) {
        xlist[idx_row] = dlist[idx_row] / blist[idx_row];
    } else if ( (idx_row-stride)<0 ) {
        int i = idx_row; int k = idx_row+stride;
        float f = clist[i]/blist[k];
        xlist[i] = (dlist[i]-dlist[k]*f)/(blist[i]-alist[k]*f);
    } else {
        int i = idx_row - stride;
        int k = idx_row;
        float f = alist[k]/blist[i];
        xlist[k] = (dlist[k]-dlist[i]*f)/(blist[k]-clist[i]*f);
    }

}

__host__
int main() {

    // Open a CSV file to save the execution times
    FILE *fp = fopen("execution_times.csv", "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file for writing.\n");
        return -1;
    }
    fprintf(fp, "System Size,Execution Time (ms)\n"); // Write the CSV header

    int sizes[] = {2, 5, 10, 100, 1000, 2000, 5000, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);  // Number of different system sizes

    int num_systems = 1000;
    for (int idx = 0; idx < num_sizes; idx++) {
        int n = sizes[idx];  // System size for this iteration
        printf("System size: %d\n", n);
        srand(time(NULL));
        float totaltime = 0;
        for (int i = 0; i <= num_systems; i++) {
            float *h_a = (float *) malloc(n * sizeof(float));  // Upper diagonal
            float *h_b = (float *) malloc(n * sizeof(float));  // Main diagonal
            float *h_c = (float *) malloc(n * sizeof(float));  // Lower diagonal
            float *h_d = (float *) malloc(n * sizeof(float));  // Right-hand side vector
            float h_x[n];

            generate_tridiagonal_system(n, h_a, h_b, h_c, h_d);

            // Device arrays
            float *d_a, *d_b, *d_c, *d_d, *d_x;

            // Allocate memory on the device
            cudaMalloc(&d_a, n * sizeof(float));
            cudaMalloc(&d_b, n * sizeof(float));
            cudaMalloc(&d_c, n * sizeof(float));
            cudaMalloc(&d_d, n * sizeof(float));
            cudaMalloc(&d_x, n * sizeof(float));

            // Copy data from host to device
            cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_c, h_c, n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_d, h_d, n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

            // Define kernel launch configuration
            int threadsPerBlock = 256;
            int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

            // Setting up timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Launch the kernel
            cudaEventRecord(start);
            Solve_Kernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, d_d, d_x, 20, n);

            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            //printf("Execution time: %f milliseconds\n", milliseconds);
            //fprintf(fp, "Execution time for system %d: %f milliseconds\n", i, milliseconds);
            if(i != 0) {
                totaltime += milliseconds;
            }

            // Copy results back to host
            cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
            cudaFree(d_d);
            cudaFree(d_x);

            free(h_a);
            free(h_b);
            free(h_c);
            free(h_d);

            // Destroy the events
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        printf("Average runtime: %f\n", totaltime/num_systems);
        fprintf(fp,"%d,%f\n", n,totaltime/num_systems);
    }

    fclose(fp);
    return 0;
}