#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__
void solve_2by2_x(float a1, float b1, float c1, float a2, float b2, float c2, float *x) {
    double det = a1 * b2 - a2 * b1;

    if (det == 0) {
        //skip
    } else {
        // Calculate the solutions using Cramer's rule
        *x = (c1 * b2 - c2 * b1) / det;
    }
}

__device__
void solve_2by2_y(float a1, float b1, float c1, float a2, float b2, float c2, float *y) {
    double det = a1 * b2 - a2 * b1;

    if (det == 0) {
        //skip
    } else {
        *y = (a1 * c2 - a2 * c1) / det;
    }
}

__device__
uint32_t flp2 (uint32_t x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}

__device__
void printall(float *a, float *b, float *c, float *d, int n) {
    printf("\na: \n");
    for (int i = 0; i < n; i++) {
        printf("%f, ", a[i]);
    }
    printf("\nb: \n");
    for (int i = 0; i < n; i++) {
        printf("%f, ", b[i]);
    }
    printf("\nc: \n");
    for (int i = 0; i < n; i++) {
        printf("%f, ", c[i]);
    }
    printf("\nd: \n");
    for (int i = 0; i < n; i++) {
        printf("%f, ", d[i]);
    }
}

__global__ void Solve_Kernel(
    float * alist, float * blist, float * clist, float * dlist, float * xlist, int iter_max, int DMax, int n) {

    int idx_row = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("block id is: %d, block dim is: %d\n", blockIdx.x, blockDim.x);
    if (idx_row >= n) {
        //printf("idx_row is: %d\n", idx_row);
        return;
    }
    int row_max = DMax - 1;

    int stride = 1;
    int next_stride = stride;

    float a1, b1, c1, d1;
    float k01, k21, c01, a21, d01, d21;

    bool next_or_ot = true;

    float i_end = log2f(n);
    uint32_t closest_power = flp2(n);
    
    for (int i = 0; i < i_end - 1; ++i) {
        //printf("iteration : %d, thread is : %d, next or not is: %d, stride is: %d\n", i, idx_row, next_or_ot, stride);
        if ( next_or_ot ) { 

            next_stride = stride << 1;
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

            //stride = next_stride;

            //Determine if this line has reached the bi-set
            //int pos = idx_row-2*stride;
            //accum = 0;
            if ((idx_row - stride) < 0 || (idx_row + stride) >= n) {
                next_or_ot = false;
                
                if ((idx_row - 2*stride) >=0 || (idx_row + 2*stride) < n) {
                    next_or_ot = true;
                }
            }
            //printf("continue? : %d\n", next_or_ot);
            if (next_or_ot) stride = next_stride;
            
        }

        // print
        __syncthreads();


        alist[idx_row] = a1;
        blist[idx_row] = b1;
        clist[idx_row] = c1;
        dlist[idx_row] = d1;
    }

    // Solve!

    /*
    if (idx_row == 0) {
        printall(alist, blist, clist, dlist, n);
    }
    */

    // When N is odd, we have one row left out
    if (idx_row >= (n - closest_power) && idx_row < (closest_power))   {
        //printf("Filled in one row with case 1\n This index is : %d\n", idx_row);
        xlist[idx_row] = dlist[idx_row]/blist[idx_row];
    }
    // When N is even, all reductions will leave us with 2 by 2 matrices
    else {
        if (idx_row <  n/2) {
            //printf("Filled in one row with case 3\n This index is : %d, with stride: %d\n. Inputs to solver are: b[idx]: %f, c[idx]: %f, d[idx]: %f, a[idx+stride]: %f, b[idx+stride]: %f, d[idx+stride]: %f\n", idx_row, stride, blist[idx_row], clist[idx_row], dlist[idx_row], alist[idx_row + stride], blist[idx_row + stride], dlist[idx_row + stride]);
            
            solve_2by2_x(blist[idx_row], clist[idx_row], dlist[idx_row], alist[idx_row + stride], blist[idx_row + stride], dlist[idx_row + stride] , &xlist[idx_row]);
        }
        else {
            //printf("Filled in one row with case 4\n This index is : %d, with stride: %d\n. Inputs to solver are: b[idx-stride]: %f, c[idx-stride]: %f, d[idx-stride]: %f, a[idx]: %f, b[idx]: %f, d[idx]: %f\n", idx_row, stride, blist[idx_row - stride], clist[idx_row - stride], dlist[idx_row - stride], alist[idx_row], blist[idx_row], dlist[idx_row]);
            solve_2by2_y(blist[idx_row - stride], clist[idx_row - stride], dlist[idx_row - stride], alist[idx_row], blist[idx_row], dlist[idx_row] , &xlist[idx_row]);
        }
    }
}


__host__
int old_main() {
    int n = 3;
    float h_a[] = {0, -1, -1};
    float h_b[] = {2, 2, 2};
    float h_c[] = {-1, -1, 0};
    float h_d[] = {0, 0, 1};
    float h_x[n];


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

    // Define kernel launch configuration
    int threadsPerBlock = 512;
    int blocks = 3;

    // Launch the kernel
    Solve_Kernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, d_d, d_x, 10, n, n);

    // Copy results back to host
    cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d, d_d, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

    /*
    printf("\n\nafter computation: \n\n");

     // Output the results
    printf("\na vector x:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", h_a[i]);
    }
     // Output the results
    printf("\nb vector x:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", h_b[i]);
    }
     // Output the results
    printf("\nc vector x:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", h_c[i]);
    }
     // Output the results
    printf("\nd vector x:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", h_d[i]);
    }

    // Output the results
    printf("\nSolution vector x:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", h_x[i]);
    }

    printf("\n");
    */

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_x);

    return 0;
}

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

__host__
void print_to_fp(int n, float diag[], FILE *fp) {
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%f, ", diag[i]);
    }
}

__host__
int test_main() {
        // Open a CSV file to save the execution times
    FILE *fp = fopen("execution_times.csv", "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file for writing.\n");
        return -1;
    }
    fprintf(fp, "System Size,Execution Time (ms)\n"); // Write the CSV header

    int sizes[] = {1025};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);  // Number of different system sizes

    int num_systems = 0;
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
            fprintf(fp, "Diagonal A: \n");
            print_to_fp(n, h_a, fp);
            fprintf(fp, "\n");
            fprintf(fp, "Diagonal B: \n");
            print_to_fp(n, h_b, fp);
            fprintf(fp, "\n");
            fprintf(fp, "Diagonal C: \n");
            print_to_fp(n, h_c, fp);
            fprintf(fp, "\n");
            fprintf(fp, "Diagonal D: \n");
            print_to_fp(n, h_d, fp);
            fprintf(fp, "\n");

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
            int threadsPerBlock = 512;
            int blocks = 3;

            // Setting up timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Launch the kernel
            cudaEventRecord(start);
            // Launch the kernel
            Solve_Kernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, d_d, d_x, 30, n, n);

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

            // Output the results
            /*
            printf("Solution vector x:\n");
            fprintf(fp, "Solution vector for system %d:\n", i);
            for (int i = 0; i < n; i++) {
                printf("%f, ", h_x[i]);
                fprintf(fp, "%f, ", h_x[i]);
            }

            printf("\n\n");
            fprintf(fp, "\n\n");      
            */      

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
            int threadsPerBlock = 512;
            int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

            // Setting up timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Launch the kernel
            cudaEventRecord(start);
            // Launch the kernel
            Solve_Kernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, d_d, d_x, 10, n, n);

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

            // Output the results
            
            /*
            printf("Solution vector x:\n");
            fprintf(fp, "Solution vector for system %d:\n", i);
            for (int i = 0; i < n; i++) {
                printf("%f, ", h_x[i]);
                fprintf(fp, "%f ", h_x[i]);
            }

            printf("\n\n");
            fprintf(fp, "\n\n");
            */
            

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


