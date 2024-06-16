#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 4 // block matrix size
#define BLOCK_SIZE 2 // m : block size 

__global__ void list_print(int nmax, float * in) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    printf("Thread %i shows %f \n", i, in[i]);
}

__device__
void print2(float *vec) {
    for (int i = 0; i < 2; i++) {
        printf("%f\n", vec[i]);
    }
}

__device__
void print4(float *mat) {
    for (int i = 0; i < 4; i++) {
        printf("%f\n", mat[i]);
    }
}

__device__
void print12(float *mat) {
    for (int i = 0; i < 12; i++) {
        printf("%f\n", mat[i]);
    }
}

__device__
void solve_2by2(float a1, float b1, float c1, float a2, float b2, float c2, float *x, float *y) {
    double det = a1 * b2 - a2 * b1;

    if (det == 0) {
        //skip
    } else {
        // Calculate the solutions using Cramer's rule
        *x = (c1 * b2 - c2 * b1) / det;
        *y = (a1 * c2 - a2 * c1) / det;
        printf("Solution 2 by 2 is; [x: %f, y: %f]\n", *x, *y);
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

__device__
void solve_4by4_x(float *A, float *B, float *C, float *D, float *K1, float *K2, float *x1) {
    float inv[BLOCK_SIZE * BLOCK_SIZE], BinvD[BLOCK_SIZE * BLOCK_SIZE], temp[BLOCK_SIZE * BLOCK_SIZE], M[BLOCK_SIZE * BLOCK_SIZE], L[BLOCK_SIZE];
    float vec_temp[BLOCK_SIZE];

    /*
    printf("A: ");
    print4(A);
    printf("\n");

    printf("B: ");
    print4(B);
    printf("\n");

    printf("C: ");
    print4(C);
    printf("\n");

    printf("D: ");
    print4(D);
    printf("\n");

    printf("K1: ");
    print2(K1);
    printf("\n");

    printf("K2: ");
    print2(K2);
    printf("\n");
    */


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



__device__
void solve_4by4_y(float *A, float *B, float *C, float *D, float *K1, float *K2, float *x2) {
    float inv[BLOCK_SIZE * BLOCK_SIZE], BinvD[BLOCK_SIZE * BLOCK_SIZE], temp[BLOCK_SIZE * BLOCK_SIZE], M[BLOCK_SIZE * BLOCK_SIZE], L[BLOCK_SIZE];
    float vec_temp[BLOCK_SIZE], vec_temp2[BLOCK_SIZE];
    float x_temp[BLOCK_SIZE];

    /*
    printf("A: ");
    print4(A);
    printf("\n");

    printf("B: ");
    print4(B);
    printf("\n");

    printf("C: ");
    print4(C);
    printf("\n");

    printf("D: ");
    print4(D);
    printf("\n");

    printf("K1: ");
    print2(K1);
    printf("\n");

    printf("K2: ");
    print2(K2);
    printf("\n");
    */

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

// Function to multiply two matrices
__device__
void multiply(double mat1[N][N], double mat2[N], double res[N]) {
    for (int i = 0; i < N; i++) {
        res[i] = 0;
        for (int j = 0; j < N; j++) {
            res[i] += mat1[i][j] * mat2[j];
        }
    }
}



__global__ void Solve_Block_Tridiagonal(
    float * a, float * b, float * c, float * d, float * x, int iter_max, int DMax) {

    int idx_row = blockIdx.x*blockDim.x + threadIdx.x;
    int row_max = DMax - 1;

    int stride = 1;
    int next_stride = stride;

    float a1[BLOCK_SIZE * BLOCK_SIZE], b1[BLOCK_SIZE * BLOCK_SIZE], c1[BLOCK_SIZE * BLOCK_SIZE], d1[BLOCK_SIZE];
    float k01[BLOCK_SIZE * BLOCK_SIZE], k21[BLOCK_SIZE * BLOCK_SIZE], c01[BLOCK_SIZE * BLOCK_SIZE], a21[BLOCK_SIZE * BLOCK_SIZE], d01[BLOCK_SIZE], d21[BLOCK_SIZE];
    float inv_b_str[BLOCK_SIZE * BLOCK_SIZE];

    bool next_or_ot = true;

    float i_end = log2f(N);


    for (int i = 0; i < i_end - 1; ++i) {
        printf("iteration : %d, thread is : %d, next or not is: %d\n", i, idx_row, next_or_ot);
        if ( next_or_ot ) { 

            next_stride = stride << 1;

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
                multiply_matrix_2by2(inv_b_str, &a[idx_row * BLOCK_SIZE * BLOCK_SIZE], k01);

                // c01 = k01*c[i-1]
                multiply_matrix_2by2(k01, &c[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], c01);
                // d01 = k01*d[i-1]
                multiply_matrix_vector_2x2(k01, &d[(idx_row - stride) * BLOCK_SIZE], d01);
            } else {
                // k01 = (b[i-1])^-1*a[i]
                invert_matrix_2x2(&b[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], inv_b_str);
                multiply_matrix_2by2(inv_b_str, &a[idx_row * BLOCK_SIZE * BLOCK_SIZE], k01);

                // a1 = -(k01*a[i-1])
                multiply_matrix_2by2(k01, &a[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], a1);
                multiply_neg_2by2(a1);

                // c01 = k01*c[i-1]
                multiply_matrix_2by2(k01, &c[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], c01);
                // d01 = k01*d[i-1]
                multiply_matrix_vector_2x2(k01, &d[(idx_row - stride) * BLOCK_SIZE], d01);
            }

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
                multiply_matrix_2by2(inv_b_str, &c[idx_row * BLOCK_SIZE * BLOCK_SIZE], k21);

                // a21 = k21*a[i+1]
                multiply_matrix_2by2(k21, &a[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], a21);
                // d21 = k21*d[i+1]
                multiply_matrix_vector_2x2(k21, &d[(idx_row + stride) * BLOCK_SIZE], d21);
            } else {
                // k21 = (b[i+1])^-1*c[i]
                invert_matrix_2x2(&b[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], inv_b_str);
                multiply_matrix_2by2(inv_b_str, &c[idx_row * BLOCK_SIZE * BLOCK_SIZE], k21);

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

            //Determine if this line has reached the bi-set
            //int pos = idx_row-2*stride;
            //accum = 0;
            if ((idx_row - stride) < 0 || (idx_row + stride) >= N) {
                next_or_ot = false;
                if ((idx_row - 2*stride) >=0 || (idx_row + 2*stride) < N) {
                    next_or_ot = true;
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

    /*
    printf("a: ");
    print12(a);
    printf("\n");

    printf("b: ");
    print12(b);
    printf("\n");

    printf("c: ");
    print12(c);
    printf("\n");

    printf("d: ");
    print12(d);
    printf("\n");
    */

    // Solve!
    
    // 2 by 2
    if ((N & 1) && idx_row == int (N/2))   {
        printf("Thread is: %d, Solving 2 by 2: case 1\n", idx_row);
        solve_2by2(b[idx_row * BLOCK_SIZE * BLOCK_SIZE], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 1], d[idx_row * BLOCK_SIZE], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 2], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 3], d[idx_row * BLOCK_SIZE + 1], &x[idx_row * BLOCK_SIZE], &x[idx_row * BLOCK_SIZE + 1]);
        //xlist[idx_row] = dlist[idx_row]/blist[idx_row];
    }
    else if (N > 2 && (N & 3) == 2 && (idx_row == int (N/2) || idx_row == int (N/2) - 1)) {
        printf("Thread is: %d, Solving 2 by 2: case 2\n", idx_row);
        solve_2by2(b[idx_row * BLOCK_SIZE * BLOCK_SIZE], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 1], d[idx_row * BLOCK_SIZE], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 2], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 3], d[idx_row * BLOCK_SIZE + 1], &x[idx_row * BLOCK_SIZE], &x[idx_row * BLOCK_SIZE + 1]);
        //xlist[idx_row] = dlist[idx_row]/blist[idx_row];
    }
    else {
        // 4 by 4
        if (idx_row < N/2) {
            // identify A, B, C, D, and K matrix
            printf("case 1\n");
            solve_4by4_x(&b[idx_row * BLOCK_SIZE * BLOCK_SIZE], &c[idx_row * BLOCK_SIZE * BLOCK_SIZE], &a[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], &b[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], &d[idx_row * BLOCK_SIZE], &d[(idx_row + stride) * BLOCK_SIZE], &x[idx_row * BLOCK_SIZE]);
        }
        else {
            printf("case 2\n");
            solve_4by4_y(&b[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], &c[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], &a[idx_row * BLOCK_SIZE * BLOCK_SIZE], &b[idx_row * BLOCK_SIZE * BLOCK_SIZE], &d[(idx_row - stride) * BLOCK_SIZE], &d[idx_row * BLOCK_SIZE], &x[idx_row * BLOCK_SIZE]);
        }
    }
}

__host__ 
void generate_block_tridiagonal_system(int n, float *h_a, float *h_b, float *h_c, float *h_d) {
    float src_a[] = {0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,  -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0};
    float src_b[] = {2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0};
    float src_c[] = {-1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0};
    float src_d[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    /*
    float src_a[] = {0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0};
    float src_b[] = {2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0};
    float src_c[] = {-1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0};
    float src_d[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    */

    memcpy(h_a, src_a, sizeof(src_a));
    memcpy(h_b, src_b, sizeof(src_b));
    memcpy(h_c, src_c, sizeof(src_c));
    memcpy(h_d, src_d, sizeof(src_d));
}

__host__
int main() {

    // Host arrays
    float *h_a = (float *) malloc(N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));  // Upper diagonal
    float *h_b = (float *) malloc(N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));  // Main diagonal
    float *h_c = (float *) malloc(N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));  // Lower diagonal
    float *h_d = (float *) malloc(N * BLOCK_SIZE * sizeof(float));  // Right-hand side vector
    float *h_x = (float *) malloc(N * BLOCK_SIZE * sizeof(float));  // Solution vector

    generate_block_tridiagonal_system(N, h_a, h_b, h_c, h_d);

    printf("\nBefore computation: \n\n");

     // Output the results
    printf("\na vector x:\n");
    for (int i = 0; i < N * BLOCK_SIZE * BLOCK_SIZE; i++) {
        printf("%f ", h_a[i]);
    }
     // Output the results
    printf("\nb vector x:\n");
    for (int i = 0; i < N * BLOCK_SIZE * BLOCK_SIZE; i++) {
        printf("%f ", h_b[i]);
    }
     // Output the results
    printf("\nc vector x:\n");
    for (int i = 0; i < N * BLOCK_SIZE * BLOCK_SIZE; i++) {
        printf("%f ", h_c[i]);
    }
     // Output the results
    printf("\nd vector x:\n");
    for (int i = 0; i < N * BLOCK_SIZE; i++) {
        printf("%f ", h_d[i]);
    }

    // Output the results
    printf("\nSolution vector x:\n");
    for (int i = 0; i < N * BLOCK_SIZE; i++) {
        printf("%f ", h_x[i]);
    }

    printf("\n");

    // Device arrays
    float *d_a, *d_b, *d_c, *d_d, *d_x;

    // Allocate memory on the device
    cudaMalloc(&d_a, N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
    cudaMalloc(&d_b, N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
    cudaMalloc(&d_c, N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
    cudaMalloc(&d_d, N * BLOCK_SIZE * sizeof(float));
    cudaMalloc(&d_x, N * BLOCK_SIZE * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, N * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Define kernel launch configuration
    int threadsPerBlock = N;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    Solve_Block_Tridiagonal<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, d_d, d_x, 20, N);

    // Copy results back to host
    cudaMemcpy(h_a, d_a, N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d, d_d, N * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x, d_x, N * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n\nafter computation: \n\n");

     // Output the results
    printf("\na vector x:\n");
    for (int i = 0; i < N * BLOCK_SIZE * BLOCK_SIZE; i++) {
        printf("%f ", h_a[i]);
    }
     // Output the results
    printf("\nb vector x:\n");
    for (int i = 0; i < N * BLOCK_SIZE * BLOCK_SIZE; i++) {
        printf("%f ", h_b[i]);
    }
     // Output the results
    printf("\nc vector x:\n");
    for (int i = 0; i < N * BLOCK_SIZE * BLOCK_SIZE; i++) {
        printf("%f ", h_c[i]);
    }
     // Output the results
    printf("\nd vector x:\n");
    for (int i = 0; i < N * BLOCK_SIZE; i++) {
        printf("%f ", h_d[i]);
    }

    // Output the results
    printf("\nSolution vector x:\n");
    for (int i = 0; i < N * BLOCK_SIZE; i++) {
        printf("%f ", h_x[i]);
    }

    printf("\n");

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

    return 0;
}