#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#define BLOCK_SIZE 2 // m : block size 

__device__
void solve_2by2(float a1, float b1, float c1, float a2, float b2, float c2, float *x, float *y) {
    double det = a1 * b2 - a2 * b1;

    if (det == 0) {
        //printf("Determinant is zero.\n");
    } else {
        // Calculate the solutions using Cramer's rule
        *x = (c1 * b2 - c2 * b1) / det;
        *y = (a1 * c2 - a2 * c1) / det;
    }
}

__device__ void invert_matrix_2x2(float *mat, float *inv_mat) {
    float determinant = mat[0] * mat[3] - mat[1] * mat[2];

    if (determinant == 0) {
      //printf("Matrix is singular and cannot be inverted.\n");
      return;
    }

    float inv_det = 1.0f / determinant;

    inv_mat[0] = inv_det * mat[3];
    inv_mat[1] = -inv_det * mat[1];
    inv_mat[2] = -inv_det * mat[2];
    inv_mat[3] = inv_det * mat[0];
}

__device__ 
void multiply_matrix_2by2(float *mat1, float *mat2, float *result) {
    // Result matrix should be initialized to zeros
    for (int i = 0; i < 4; ++i) {
        result[i] = 0.0;
    }

    // Perform matrix multiplication
    result[0] = mat1[0] * mat2[0] + mat1[1] * mat2[2];
    result[1] = mat1[0] * mat2[1] + mat1[1] * mat2[3];
    result[2] = mat1[2] * mat2[0] + mat1[3] * mat2[2];
    result[3] = mat1[2] * mat2[1] + mat1[3] * mat2[3];
}

__device__
void multiply_matrix_vector_2x2(float *mat, float *vec, float *result) {
    // Perform matrix-vector multiplication
    result[0] = mat[0] * vec[0] + mat[1] * vec[1];
    result[1] = mat[2] * vec[0] + mat[3] * vec[1];
}

__device__
void multiply_neg_2by2(float *mat) {
    for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
        mat[i] = -mat[i];
    }
}

__device__
void add_vectors_2x2(float *vec1, float *vec2, float *result) {
    for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
      result[i] = vec1[i] + vec2[i];
    }
}

__device__
void subtract_vectors_2x2(float *vec1, float *vec2, float *result) {
    result[0] = vec1[0] - vec2[0];
    result[1] = vec1[1] - vec2[1];
}

__device__
void subtract_2x2(float *mat1, float *mat2, float *result) {
    result[0] = mat1[0] - mat2[0];
    result[1] = mat1[1] - mat2[1];
    result[2] = mat1[2] - mat2[2];
    result[3] = mat1[3] - mat2[3];
}

// Solver for 4 by 4 matrix system by partitioning into 4 2 by 2 blocks. K1 and K2 are 2 by 1 vectors of RHS. x variant fills in x1 (2 by 1)
__device__
void solve_4by4_x(int idx_row, int stride, float *A, float *B, float *C, float *D, float *K1, float *K2, float *x1) {
    float inv[BLOCK_SIZE * BLOCK_SIZE], BinvD[BLOCK_SIZE * BLOCK_SIZE], temp[BLOCK_SIZE * BLOCK_SIZE], M[BLOCK_SIZE * BLOCK_SIZE], L[BLOCK_SIZE];
    float vec_temp[BLOCK_SIZE];

    // x1 [2x1]
    // factor 1
    invert_matrix_2x2(D, inv);
    multiply_matrix_2by2(B, inv, BinvD);
    // inv free
    multiply_matrix_2by2(BinvD, C, temp);
    subtract_2x2(A, temp, M);
    // temp1 free

    // factor 2
    multiply_matrix_vector_2x2(BinvD, K2, vec_temp);
    subtract_vectors_2x2(K1, vec_temp, L);

    // invert M and multiply
    invert_matrix_2x2(M, inv);
    multiply_matrix_vector_2x2(inv, L, x1);
    
}

// Solver for 4 by 4 matrix system by partitioning into 4 2 by 2 blocks. K1 and K2 are 2 by 1 vectors of RHS. y variant fills in x2 (2 by 1)
__device__
void solve_4by4_y(int idx_row, int stride, float *A, float *B, float *C, float *D, float *K1, float *K2, float *x2) {
    float inv[BLOCK_SIZE * BLOCK_SIZE], BinvD[BLOCK_SIZE * BLOCK_SIZE], temp[BLOCK_SIZE * BLOCK_SIZE], M[BLOCK_SIZE * BLOCK_SIZE], L[BLOCK_SIZE];
    float vec_temp[BLOCK_SIZE], vec_temp2[BLOCK_SIZE];
    float x_temp[BLOCK_SIZE];
    
    // x1 [2x1]
    // factor 1
    invert_matrix_2x2(D, inv);
    multiply_matrix_2by2(B, inv, BinvD);
    // inv free
    multiply_matrix_2by2(BinvD, C, temp);
    subtract_2x2(A, temp, M);
    // temp1 free

    // factor 2
    multiply_matrix_vector_2x2(BinvD, K2, vec_temp);
    subtract_vectors_2x2(K1, vec_temp, L);

    // invert M and multiply
    invert_matrix_2x2(M, inv);
    multiply_matrix_vector_2x2(inv, L, x_temp);
    

    // x2 [2x1]
    invert_matrix_2x2(D, inv);
    multiply_matrix_vector_2x2(C, x_temp, vec_temp);
    subtract_vectors_2x2(K2, vec_temp, vec_temp2);
    multiply_matrix_vector_2x2(inv, vec_temp2, x2);
}

__global__ void Solve_Block_Tridiagonal(
    float * a, float * b, float * c, float * d, float * x, int iter_max, int DMax, int n) {

    int idx_row = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx_row >= n) return;
    int row_max = DMax - 1;

    int stride = 1;
    int next_stride = stride;

    float a1[BLOCK_SIZE * BLOCK_SIZE], b1[BLOCK_SIZE * BLOCK_SIZE], c1[BLOCK_SIZE * BLOCK_SIZE], d1[BLOCK_SIZE];
    float k01[BLOCK_SIZE * BLOCK_SIZE], k21[BLOCK_SIZE * BLOCK_SIZE], c01[BLOCK_SIZE * BLOCK_SIZE], a21[BLOCK_SIZE * BLOCK_SIZE], d01[BLOCK_SIZE], d21[BLOCK_SIZE];
    float inv_b_str[BLOCK_SIZE * BLOCK_SIZE];

    bool next_or_not = true;

    float i_end = log2f(n);
    float closest_power = (float) (int) i_end;

    for (int i = 0; i <= i_end - 1; ++i) {
        // Check to see if we've reached reduction limit 
        if ( next_or_not ) { 
            next_stride = stride << 1;

            // Reduce rows

            // Compute coefficients for row below
            if ((idx_row - stride) < 0) {
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
                    a1[i] = 0.0f;
                    k01[i] = 0.0f;
                    c01[i] = 0.0f;
                }
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    d01[i] = 0.0f;
                }
            } else if ((idx_row - next_stride) < 0) {
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) a1[i] = 0.0f;

                // k01 = (b[i-1])^-1*a[i]
                invert_matrix_2x2(&b[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], inv_b_str);
                multiply_matrix_2by2(&a[idx_row * BLOCK_SIZE * BLOCK_SIZE], inv_b_str, k01);

                // c01 = k01*c[i-1]
                multiply_matrix_2by2(k01, &c[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], c01);
                // d01 = k01*d[i-1]
                multiply_matrix_vector_2x2(k01, &d[(idx_row - stride) * BLOCK_SIZE], d01);
            } else {
                // k01 = (b[i-1])^-1*a[i]
                invert_matrix_2x2(&b[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], inv_b_str);
                multiply_matrix_2by2(&a[idx_row * BLOCK_SIZE * BLOCK_SIZE], inv_b_str, k01);

                // a1 = -(k01*a[i-1])
                multiply_matrix_2by2(k01, &a[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], a1);
                multiply_neg_2by2(a1);

                // c01 = k01*c[i-1]
                multiply_matrix_2by2(k01, &c[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], c01);
                // d01 = k01*d[i-1]
                multiply_matrix_vector_2x2(k01, &d[(idx_row - stride) * BLOCK_SIZE], d01);
            }

            // Compute coefficients for row below
            if ((idx_row + stride) > row_max) {
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
                    c1[i] = 0.0f;
                    k21[i] = 0.0f;
                    a21[i] = 0.0f;
                }
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    d21[i] = 0.0f;
                }
            } else if ((idx_row + next_stride) > row_max) {
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) c1[i] = 0.0f;

                // k21 = (b[i+1])^-1*c[i]
                invert_matrix_2x2(&b[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], inv_b_str);
                multiply_matrix_2by2(&c[idx_row * BLOCK_SIZE * BLOCK_SIZE], inv_b_str, k21);

                // a21 = k21*a[i+1]
                multiply_matrix_2by2(k21, &a[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], a21);
                // d21 = k21*d[i+1]
                multiply_matrix_vector_2x2(k21, &d[(idx_row + stride) * BLOCK_SIZE], d21);
            } else {
                // k21 = (b[i+1])^-1*c[i]
                invert_matrix_2x2(&b[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], inv_b_str);
                multiply_matrix_2by2(&c[idx_row * BLOCK_SIZE * BLOCK_SIZE], inv_b_str, k21);

                // c1 = -(k21*c[i+1])
                multiply_matrix_2by2(k21, &c[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], c1);
                multiply_neg_2by2(c1);

                // a21 = k21*a[i+1]
                multiply_matrix_2by2(k21, &a[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], a21);
                // d21 = k21*d[i+1]
                multiply_matrix_vector_2x2(k21, &d[(idx_row + stride) * BLOCK_SIZE], d21);
            }

            for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
                b1[i] = b[idx_row * BLOCK_SIZE * BLOCK_SIZE + i] - c01[i] - a21[i];
            }
            for (int i = 0; i < BLOCK_SIZE; i++) {
                d1[i] = d[idx_row * BLOCK_SIZE + i] - d01[i] - d21[i];
            }

            stride = next_stride;
            
            // Determine reduction at limit
            if ((idx_row - next_stride) < 0 || (idx_row + next_stride) >= n) {
                next_or_not = false;
                
                if ((idx_row - 2*next_stride) >=0 || (idx_row + 2*next_stride) < n) {
                    next_or_not = true;
                }
            }
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
            a[idx_row * BLOCK_SIZE * BLOCK_SIZE + i] = a1[i];
            b[idx_row * BLOCK_SIZE * BLOCK_SIZE + i] = b1[i];
            c[idx_row * BLOCK_SIZE * BLOCK_SIZE + i] = c1[i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            d[idx_row * BLOCK_SIZE + i] = d1[i];
        }
    }
    
    // Solve!

    // 2 by 2 to solve for middle sector in the case of non power of 2 rows.
    if (idx_row >= (n - closest_power) && idx_row < (closest_power)) {
        solve_2by2(b[idx_row * BLOCK_SIZE * BLOCK_SIZE], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 1], d[idx_row * BLOCK_SIZE], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 2], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 3], d[idx_row * BLOCK_SIZE + 1], &x[idx_row * BLOCK_SIZE], &x[idx_row * BLOCK_SIZE + 1]);
        //xlist[idx_row] = dlist[idx_row]/blist[idx_row];
    }
    else {
        // 4 by 4
        if (idx_row < n/2) {
            // Fill in x1
            solve_4by4_x(idx_row, stride, &b[idx_row * BLOCK_SIZE * BLOCK_SIZE], &c[idx_row * BLOCK_SIZE * BLOCK_SIZE], &a[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], &b[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], &d[idx_row * BLOCK_SIZE], &d[(idx_row + stride) * BLOCK_SIZE], &x[idx_row * BLOCK_SIZE]);
        }
        else {
            // Fill in x2
            solve_4by4_y(idx_row, stride, &b[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], &c[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], &a[idx_row * BLOCK_SIZE * BLOCK_SIZE], &b[idx_row * BLOCK_SIZE * BLOCK_SIZE], &d[(idx_row - stride) * BLOCK_SIZE], &d[idx_row * BLOCK_SIZE], &x[idx_row * BLOCK_SIZE]);
        }
    }
}


// Sample system of size n = 5
__host__ 
void generate_block_tridiagonal_system(int n, float *h_a, float *h_b, float *h_c, float *h_d) {
    float src_a[] = {0.000000, 0.000000, 0.000000, 0.000000, 75.000000, 88.000000, 62.000000, 4.000000, 78.000000, 90.000000, 35.000000, 84.000000, 75.000000, 45.000000, 48.000000, 26.000000, 84.000000, 21.000000, 97.000000, 50.000000};
    float src_b[] = {151.000000, 107.000000, 148.000000, 170.000000, 235.000000, 202.000000, 242.000000, 107.000000, 230.000000, 208.000000, 201.000000, 231.000000, 230.000000, 201.000000, 261.000000, 236.000000, 226.000000, 96.000000, 196.000000, 125.000000};
    float src_c[] = {62.000000, 3.000000, 70.000000, 69.000000, 84.000000, 58.000000, 89.000000, 29.000000, 24.000000, 49.000000, 64.000000, 83.000000, 37.000000, 18.000000, 92.000000, 80.000000, 0.000000, 0.000000, 0.000000, 0.000000};
    float src_d[] = {25.000000, 80.000000, 16.000000, 73.000000, 99.000000, 50.000000, 88.000000, 4.000000, 85.000000, 70.000000};

    memcpy(h_a, src_a, sizeof(src_a));
    memcpy(h_b, src_b, sizeof(src_b));
    memcpy(h_c, src_c, sizeof(src_c));
    memcpy(h_d, src_d, sizeof(src_d));
}

// Testing function
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
void print_to_fp(int n, float diag[], FILE *fp) {
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%f, ", diag[i]);
    }
}

__host__
int main() {

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
            Solve_Block_Tridiagonal<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, d_d, d_x, 30, n, n);

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
