#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 2

__host__
void generate_block_tridiagonal_system(int n, float a[], float b[], float c[], float d[]) {
    // Hardcode the specific block-tridiagonal matrix values
    // Upper diagonal a (entries start from index 1)
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            a[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.0f;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.0f;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.0f;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.0f;
        }
        else if (i == 1) {
            a[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.024606;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 2] = -0.320170;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }
        else if (i == 2) {
            a[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.024606;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 2] = -0.320170;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }
        else if (i == 3) {
            a[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.024606;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 2] = -0.316363;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }
        else if (i == 4) {
            a[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.024606;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 2] = -0.297475;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }
        else if (i == 5) {
            a[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.024606;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 2] = -0.253063;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }
        else if (i == 6) {
            a[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.024606;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 2] = -0.186637;
            a[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }

    }

    // Main diagonal b
    for (int i = 0; i < n; i++) {
        if (i == 0) {
          b[i * BLOCK_SIZE * BLOCK_SIZE + 0] = -0.326371;
          b[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.000000;
          b[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.000000;
          b[i * BLOCK_SIZE * BLOCK_SIZE + 3] = -0.246063;
        }
        else if (i == 1) {
            b[i * BLOCK_SIZE * BLOCK_SIZE + 0] = -0.655202;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.295563;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.295563;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 3] = -0.806212;
        }
        else if (i == 2) {
            b[i * BLOCK_SIZE * BLOCK_SIZE + 0] = -0.655202;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.295563;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.295563;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 3] = -0.806298;
        }
        else if (i == 3) {
            b[i * BLOCK_SIZE * BLOCK_SIZE + 0] = -0.655202;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.291757;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.291757;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 3] = -0.798788;
        }
        else if (i == 4) {
            b[i * BLOCK_SIZE * BLOCK_SIZE + 0] = -0.655202;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.272869;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.272869;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 3] = -0.824240;
        }
        else if (i == 5) {
            b[i * BLOCK_SIZE * BLOCK_SIZE + 0] = -0.655202;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.228457;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.228457;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 3] = -0.749323;
        }
        else if (i == 6) {
            b[i * BLOCK_SIZE * BLOCK_SIZE + 0] = -0.338825;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.162030;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.162030;
            b[i * BLOCK_SIZE * BLOCK_SIZE + 3] = -0.363043;
        }
    }

    // Lower diagonal c
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            c[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 1] = -0.320170;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.024606;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }
        else if (i == 1) {
            c[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 1] = -0.320170;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.024606;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }
        else if (i == 2) {
            c[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 1] = -0.316363;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.024606;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }
        else if (i == 3) {
            c[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 1] = -0.297475;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.024606;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }
        else if (i == 4) {
            c[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 1] = -0.253063;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.024606;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }
        else if (i == 5) {
            c[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.326371;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 1] = -0.186637;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.024606;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.246063;
        }
        else if (i == 6) {
            c[i * BLOCK_SIZE * BLOCK_SIZE + 0] = 0.0f;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 1] = 0.0f;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 2] = 0.0f;
            c[i * BLOCK_SIZE * BLOCK_SIZE + 3] = 0.0f;
        }
    }

    // Right-hand side vector d
    d[0] = 3.075969;
    d[1] = 0.000000;
    d[2] = -0.000000;
    d[3] = 1.498219;
    d[4] = 0.000793;
    d[5] = 2.337883;
    d[6] = 0.001153;
    d[7] = 2.282602;
    d[8] = 0.001442;
    d[9] = 2.654664;
    d[10] = 0.001486;
    d[11] = 2.700398;
    d[12] = 0.041852;
    d[13] = 2.683804;
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

__device__ void multiply_matrix_vector_2x2(float *mat, float *vec, float *result) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
      result[i] = 0;
      for (int j = 0; j < BLOCK_SIZE; j++) {
        result[i] += mat[i * BLOCK_SIZE + j] * vec[j];
      }
    }
}

__device__ void add_vectors_2x2(float *vec1, float *vec2, float *result) {
    for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
      result[i] = vec1[i] + vec2[i];
    }
}

__device__ void subtract_vectors_2x2(float *vec1, float *vec2, float *result) {
    for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
      result[i] = vec1[i] - vec2[i];
    }
}

__global__ void Solve_Block_Tridiagonal(
    float *a, float *b, float *c, float *d, float *x,
    int iter_max, int DMax) {

    int idx_row = blockIdx.x * blockDim.x + threadIdx.x;
    int row_max = DMax - 1;

    int stride = 1;
    int next_stride = stride;

    float a1[BLOCK_SIZE * BLOCK_SIZE], b1[BLOCK_SIZE * BLOCK_SIZE], c1[BLOCK_SIZE * BLOCK_SIZE], d1[BLOCK_SIZE];
    float k01[BLOCK_SIZE * BLOCK_SIZE], k21[BLOCK_SIZE * BLOCK_SIZE], c01[BLOCK_SIZE * BLOCK_SIZE], a21[BLOCK_SIZE * BLOCK_SIZE], d01[BLOCK_SIZE], d21[BLOCK_SIZE];
    float inv_b_str[BLOCK_SIZE * BLOCK_SIZE];

    bool next_or_ot = true;
    int accum;

    for (int iter = 0; iter < iter_max; iter++) {

        if (next_or_ot) {

            next_stride = stride << 1;

            if ((idx_row - stride) < 0) {
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) a1[i] = 0.0f;
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) k01[i] = 0.0f;
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) c01[i] = 0.0f;
                for (int i = 0; i < BLOCK_SIZE; i++) d01[i] = 0.0f;
            } else if ((idx_row - next_stride) < 0) {
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) a1[i] = 0.0f;

                invert_matrix_2x2(&b[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], inv_b_str);
                multiply_matrix_vector_2x2(inv_b_str, &a[idx_row * BLOCK_SIZE * BLOCK_SIZE], k01);

                multiply_matrix_vector_2x2(&c[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], k01, c01);
                multiply_matrix_vector_2x2(&d[(idx_row - stride) * BLOCK_SIZE], k01, d01);
            } else {
                invert_matrix_2x2(&b[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], inv_b_str);
                multiply_matrix_vector_2x2(inv_b_str, &a[idx_row * BLOCK_SIZE * BLOCK_SIZE], k01);

                multiply_matrix_vector_2x2(&a[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], k01, c01);
                multiply_matrix_vector_2x2(&d[(idx_row - stride) * BLOCK_SIZE], k01, d01);
            }

            if ((idx_row + stride) > row_max) {
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) c1[i] = 0.0f;
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) k21[i] = 0.0f;
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) a21[i] = 0.0f;
                for (int i = 0; i < BLOCK_SIZE; i++) d21[i] = 0.0f;
            } else if ((idx_row + next_stride) > row_max) {
                for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) c1[i] = 0.0f;

                invert_matrix_2x2(&b[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], inv_b_str);
                multiply_matrix_vector_2x2(inv_b_str, &c[idx_row * BLOCK_SIZE * BLOCK_SIZE], k21);

                multiply_matrix_vector_2x2(&a[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], k21, a21);
                multiply_matrix_vector_2x2(&d[(idx_row + stride) * BLOCK_SIZE], k21, d21);
            } else {
                invert_matrix_2x2(&b[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], inv_b_str);
                multiply_matrix_vector_2x2(inv_b_str, &c[idx_row * BLOCK_SIZE * BLOCK_SIZE], k21);

                multiply_matrix_vector_2x2(&a[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], k21, a21);
                multiply_matrix_vector_2x2(&d[(idx_row + stride) * BLOCK_SIZE], k21, d21);
            }

            for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) b1[i] = b[idx_row * BLOCK_SIZE * BLOCK_SIZE + i] - c01[i] - a21[i];
            for (int i = 0; i < BLOCK_SIZE; i++) d1[i] = d[idx_row * BLOCK_SIZE + i] - d01[i] - d21[i];

            stride = next_stride;

            int pos = idx_row - 2 * stride;
            accum = 0;
            for (size_t iter = 0; iter < 5; iter++) {
                if (pos >= 0 && pos < DMax) accum++;
                pos += stride;
            }
            if (accum < 3) {
                next_or_ot = false;
            }
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) a[idx_row * BLOCK_SIZE * BLOCK_SIZE + i] = a1[i];
        for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) b[idx_row * BLOCK_SIZE * BLOCK_SIZE + i] = b1[i];
        for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) c[idx_row * BLOCK_SIZE * BLOCK_SIZE + i] = c1[i];
        for (int i = 0; i < BLOCK_SIZE; i++) d[idx_row * BLOCK_SIZE + i] = d1[i];
    }

    if (accum == 1) {
        for (int i = 0; i < BLOCK_SIZE; i++) x[idx_row * BLOCK_SIZE + i] = d[idx_row * BLOCK_SIZE + i] / b[idx_row * BLOCK_SIZE * BLOCK_SIZE + i * BLOCK_SIZE + i];
    } else if ((idx_row - stride) < 0) {
        int i = idx_row;
        int k = idx_row + stride;
        float f = c[idx_row * BLOCK_SIZE * BLOCK_SIZE + i] / b[idx_row * BLOCK_SIZE * BLOCK_SIZE + k];
        for (int j = 0; j < BLOCK_SIZE; j++) {
            x[idx_row * BLOCK_SIZE + j] = (d[idx_row * BLOCK_SIZE + j] - d[idx_row * BLOCK_SIZE + j] * f) / (b[idx_row * BLOCK_SIZE * BLOCK_SIZE + j * BLOCK_SIZE + j] - a[idx_row * BLOCK_SIZE * BLOCK_SIZE + k] * f);
        }
    } else {
        int i = idx_row - stride;
        int k = idx_row;
        float f = a[idx_row * BLOCK_SIZE * BLOCK_SIZE + k] / b[idx_row * BLOCK_SIZE * BLOCK_SIZE + i];
        for (int j = 0; j < BLOCK_SIZE; j++) {
            x[idx_row * BLOCK_SIZE + j] = (d[idx_row * BLOCK_SIZE + j] - d[idx_row * BLOCK_SIZE + j] * f) / (b[idx_row * BLOCK_SIZE * BLOCK_SIZE + j * BLOCK_SIZE + j] - c[idx_row * BLOCK_SIZE * BLOCK_SIZE + i] * f);
        }
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

    int sizes[] = {7};  // System size for this iteration
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
            float *h_x = (float *) malloc(n * BLOCK_SIZE * sizeof(float));  // Solution vector

            generate_block_tridiagonal_system(n, h_a, h_b, h_c, h_d);

            // Device arrays
            float *d_a, *d_b, *d_c, *d_d, *d_x;

            // Allocate memory on the device
            cudaMalloc(&d_a, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
            cudaMalloc(&d_b, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
            cudaMalloc(&d_c, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
            cudaMalloc(&d_d, n * BLOCK_SIZE * sizeof(float));
            cudaMalloc(&d_x, n * BLOCK_SIZE * sizeof(float));

            // Copy data from host to device
            cudaMemcpy(d_a, h_a, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, h_b, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_c, h_c, n * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_d, h_d, n * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_x, h_x, n * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);

            // Define kernel launch configuration
            int threadsPerBlock = 256;
            int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

            // Setting up timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Launch the kernel
            cudaEventRecord(start);
            Solve_Block_Tridiagonal<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, d_d, d_x, 20, n);

            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            if (i != 0) {
                totaltime += milliseconds;
            }

            // Copy results back to host
            cudaMemcpy(h_x, d_x, n * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            // Print the results
            if (i == 0) {  // Only print results for the first iteration
                for (int j = 0; j < n * BLOCK_SIZE; j++) {
                    printf("x[%d] = %f\n", j, h_x[j]);
                }
            }

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

        printf("Average runtime: %f\n", totaltime / num_systems);
        fprintf(fp, "%d,%f\n", n, totaltime / num_systems);
    }

    fclose(fp);
    return 0;
}