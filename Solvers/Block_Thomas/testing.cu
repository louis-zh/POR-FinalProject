#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 2

__device__
void matrixMultiply(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int x = 0; x < k; x++) {
                C[i * n + j] += A[i * k + x] * B[x * n + j];
            }
        }
    }
}

__device__
void matrixSubtract(float *A, float *B, float *C, int m, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            C[row * n + col] = A[row * n + col] - B[row * n + col];
        }
    }
}

__device__
void invertMatrix(float *mat, float *invMat, int n) {
    // This simple inversion is appropriate only for 2x2 matrices
    float a = mat[0], b = mat[1], c = mat[2], d = mat[3];
    float det = a * d - b * c;
    if (det == 0) {
        printf("Matrix is singular and cannot be inverted.\n");
        return;
    }
    float invDet = 1.0 / det;
    invMat[0] =  d * invDet;
    invMat[1] = -b * invDet;
    invMat[2] = -c * invDet;
    invMat[3] =  a * invDet;
}

__global__
void blockTridiagonalSolver(float *a, float *b, float *c, float *d, float *x, int N, int blockSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx != 0) return;  // Ensure only one thread does the work

    int matSize = blockSize * blockSize;
    int vecSize = blockSize;  // Simplified vector size handling
    float *cp = (float *)malloc((N-1) * matSize * sizeof(float));
    float *dp = (float *)malloc(N * vecSize * sizeof(float));
    float *A_inv = (float *)malloc(matSize * sizeof(float));
    float *temp = (float *)malloc(matSize * sizeof(float));

    // First transformation using b[0]
    invertMatrix(b, A_inv, blockSize);
    matrixMultiply(A_inv, c, cp, blockSize, blockSize, blockSize);
    matrixMultiply(A_inv, d, dp, blockSize, blockSize, 1);

    // Forward sweep
    for (int i = 1; i < N; i++) {
        matrixMultiply(a + (i-1) * matSize, cp + (i-1) * matSize, temp, blockSize, blockSize, blockSize);
        matrixSubtract(b + i * matSize, temp, temp, blockSize, blockSize);
        invertMatrix(temp, A_inv, blockSize);
        if (i < N-1) {
            matrixMultiply(A_inv, c + i * matSize, cp + i * matSize, blockSize, blockSize, blockSize);
        }
        matrixMultiply(a + (i-1) * matSize, dp + (i-1) * vecSize, temp, blockSize, blockSize, 1);
        matrixSubtract(d + i * vecSize, temp, temp, blockSize, 1);
        matrixMultiply(A_inv, temp, dp + i * vecSize, blockSize, blockSize, 1);
    }

    // Backward substitution
    memcpy(x + (N-1) * vecSize, dp + (N-1) * vecSize, vecSize * sizeof(float));
    for (int i = N-2; i >= 0; i--) {
        matrixMultiply(cp + i * matSize, x + (i+1) * vecSize, temp, blockSize, blockSize, 1);
        matrixSubtract(dp + i * vecSize, temp, x + i * vecSize, blockSize, 1);
    }

    free(cp);
    free(dp);
    free(A_inv);
    free(temp);
}


__host__
void generate_test_system(float a[], float b[], float c[], float d[], int n) {
    int total = n*BLOCK_SIZE*BLOCK_SIZE;
    for (int i = 0; i < total; i++) {
        // Generate random values for a, b, c, and d
        a[i] = (i > 3) ? rand() % 100 + 1 : 0;  // Upper diagonal (no entry at i=0)
        c[i] = (i < total-4) ? rand() % 100 + 1 : 0;  // Lower diagonal (no entry at i=n-1)
        b[i] = rand() % 100 + 1;  // Ensure diagonal dominance
    }
    for (int i = 0; i < n*BLOCK_SIZE; i++) {
        d[i] = rand() % 100 + 1;
    }
}


__host__
int main() {
    const int blockSize = 2;

    FILE *fp = fopen("execution_times.csv", "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file for writing.\n");
        return -1;
    }
    fprintf(fp, "System Size,Execution Time (ms)\n"); // Write the CSV header

    int sizes[] = {2, 5, 10, 100, 500, 750, 1000, 2000, 5000}; // Sizes of n.
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);  // Number of different system sizes

    int num_systems = 1000;
    for (int idx = 0; idx < num_sizes; idx++) {
        int n = sizes[idx];  // System size for this iteration
        printf("System size: %d\n", n);
        srand(time(NULL));
        float totaltime = 0;
        for (int i = 0; i <= num_systems; i++) {
            float *h_a = (float *) malloc(n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));  // Upper diagonal
            float *h_b = (float *) malloc(n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));  // Main diagonal
            float *h_c = (float *) malloc(n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));  // Lower diagonal
            float *h_d = (float *) malloc(n * BLOCK_SIZE * sizeof(float));  // Right-hand side vector
            float *h_x = (float *) malloc(n * BLOCK_SIZE * sizeof(float));

            generate_test_system(h_a, h_b, h_c, h_d, n);
            //generate_block_tridiagonal_system(n, h_a, h_b, h_c, h_d);

            /*
            fprintf(fp, "Diagonal A: \n");
            print_to_fp(n*BLOCK_SIZE*BLOCK_SIZE, h_a, fp);
            fprintf(fp, "\n");
            fprintf(fp, "Diagonal B: \n");
            print_to_fp(n*BLOCK_SIZE*BLOCK_SIZE, h_b, fp);
            fprintf(fp, "\n");
            fprintf(fp, "Diagonal C: \n");
            print_to_fp(n*BLOCK_SIZE*BLOCK_SIZE, h_c, fp);
            fprintf(fp, "\n");
            fprintf(fp, "Diagonal D: \n");
            print_to_fp(n*BLOCK_SIZE, h_d, fp);
            fprintf(fp, "\n");
            */
            

            // Device arrays
            float *d_a, *d_b, *d_c, *d_d, *d_x;

            // Allocate memory on the device
            cudaMalloc(&d_a, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
            cudaMalloc(&d_b, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
            cudaMalloc(&d_c, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
            cudaMalloc(&d_d, n * BLOCK_SIZE * sizeof(float));
            cudaMalloc(&d_x, n * BLOCK_SIZE * sizeof(float));

            //printf("cudaMallos done \n");

            // Copy data from host to device
            cudaMemcpy(d_a, h_a, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, h_b, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_c, h_c, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_d, h_d, n * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_x, h_x, n * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);

            //printf("cudamemcpy done \n");

            // Define kernel launch configuration
            int threadsPerBlock = 256;
            int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

            // Setting up timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Launch the kernel
            cudaEventRecord(start);
            // Launch the kernel
            blockTridiagonalSolver<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, d_d, d_x, n, blockSize);

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
            cudaMemcpy(h_x, d_x, n * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            /*
            printf("Solution vector x:\n");
            fprintf(fp, "Solution vector for system %d:\n", i);
            for (int i = 0; i < n*BLOCK_SIZE; i++) {
                printf("%f, ", h_x[i]);
                fprintf(fp, "%f, ", h_x[i]);
            }
            */

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
            free(h_x);

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
