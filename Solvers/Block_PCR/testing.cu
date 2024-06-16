#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#define BLOCK_SIZE 2 // m : block size 

__global__ void list_print(int nmax, float * in) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    printf("Thread %i shows %f \n", i, in[i]);
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
        printf("Determinant is zero.\n");
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
void solve_4by4_x(int idx_row, int stride, float *A, float *B, float *C, float *D, float *K1, float *K2, float *x1) {
    float inv[BLOCK_SIZE * BLOCK_SIZE], BinvD[BLOCK_SIZE * BLOCK_SIZE], temp[BLOCK_SIZE * BLOCK_SIZE], M[BLOCK_SIZE * BLOCK_SIZE], L[BLOCK_SIZE];
    float vec_temp[BLOCK_SIZE];

    /*
    if (idx_row == 0) {
        printf("Inputs to 4 by 4  x solver are\n");

        printf("\nA: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", A[i]);
        }
        printf("\nB: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", B[i]);
        }
        printf("\nC: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", C[i]);
        }
        printf("\nD: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", D[i]);
        }
        printf("\nK1: \n");
        for (int i = 0; i < BLOCK_SIZE; i++) {
            printf("%f, ", K1[i]);
        }
        printf("\nK2: \n");
        for (int i = 0; i < BLOCK_SIZE; i++) {
            printf("%f, ", K2[i]);
        }
    }
    */


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

    //printf("From 4x4x at idx_row: %d, stride: %d, filled x1 as: %f, %f\n", idx_row, stride, x1[0], x1[1]);

    
}



__device__
void solve_4by4_y(int idx_row, int stride, float *A, float *B, float *C, float *D, float *K1, float *K2, float *x2) {
    float inv[BLOCK_SIZE * BLOCK_SIZE], BinvD[BLOCK_SIZE * BLOCK_SIZE], temp[BLOCK_SIZE * BLOCK_SIZE], M[BLOCK_SIZE * BLOCK_SIZE], L[BLOCK_SIZE];
    float vec_temp[BLOCK_SIZE], vec_temp2[BLOCK_SIZE];
    float x_temp[BLOCK_SIZE];
    
    
    if (idx_row == 6) {
        printf("Weird..// Inputs to 4 by 4 y  solver are\n");

        printf("\nA: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", A[i]);
        }
        printf("\nB: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", B[i]);
        }
        printf("\nC: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", C[i]);
        }
        printf("\nD: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", D[i]);
        }
        printf("\nK1: \n");
        for (int i = 0; i < BLOCK_SIZE; i++) {
            printf("%f, ", K1[i]);
        }
        printf("\nK2: \n");
        for (int i = 0; i < BLOCK_SIZE; i++) {
            printf("%f, ", K2[i]);
        }
    }
    
    

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
    //printf("From 4x4y at idx_row: %d, stride: %d, filled x2 as: %f, %f\n", idx_row, stride, x2[0], x2[1]);

    /*
    if (isnan(x2[0])) {
        printf("NaN !!!!  Inputs to 4 by 4 y  solver are\n");

        printf("\nA: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", A[i]);
        }
        printf("\nB: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", B[i]);
        }
        printf("\nC: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", C[i]);
        }
        printf("\nD: \n");
        for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
            printf("%f, ", D[i]);
        }
        printf("\nK1: \n");
        for (int i = 0; i < BLOCK_SIZE; i++) {
            printf("%f, ", K1[i]);
        }
        printf("\nK2: \n");
        for (int i = 0; i < BLOCK_SIZE; i++) {
            printf("%f, ", K2[i]);
        }
    }
    */
}

__device__
void printall(float *a, float *b, float *c, float *d, int n) {
    printf("\na: \n");
    for (int i = 0; i < n*BLOCK_SIZE*BLOCK_SIZE; i++) {
        printf("%f, ", a[i]);
    }
    printf("\nb: \n");
    for (int i = 0; i < n*BLOCK_SIZE*BLOCK_SIZE; i++) {
        printf("%f, ", b[i]);
    }
    printf("\nc: \n");
    for (int i = 0; i < n*BLOCK_SIZE*BLOCK_SIZE; i++) {
        printf("%f, ", c[i]);
    }
    printf("\nd: \n");
    for (int i = 0; i < n*BLOCK_SIZE; i++) {
        printf("%f, ", d[i]);
    }
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

    bool next_or_ot = true;

    float i_end = log2f(n);
    printf("I_end is: %f\n", i_end);
    //uint32_t closest_power = flp2(n);
    float closest_power = (float) (int) i_end;
    printf("Closest poewr is: %f\n", closest_power);

    for (int i = 0; i <= i_end - 1; ++i) {
        printf("iteration : %d, thread is : %d, next or not is: %d\n", i, idx_row, next_or_ot);
        

        /*
        if ((idx_row - next_stride) < 0 || (idx_row + next_stride) >= n) {
            next_or_ot = false;
            //printf("%d has become false.\n", idx_row);
            
            if ((idx_row - 2*next_stride) >=0 || (idx_row + 2*next_stride) < n) {
                next_or_ot = true;
                //printf("%d has become true.\n", idx_row);
            }
        }
        else {
            //printf("%d stayed true.\n", idx_row);
        }
        */
        
        if ( next_or_ot ) { 
            next_stride = stride << 1;
            //stride = next_stride;

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
            //Determine if this line has reached the bi-set
            //int pos = idx_row-2*stride;
            //accum = 0;
            
            if ((idx_row - next_stride) < 0 || (idx_row + next_stride) >= n) {
                next_or_ot = false;
                //printf("%d has become false.\n", idx_row);
                
                if ((idx_row - 2*next_stride) >=0 || (idx_row + 2*next_stride) < n) {
                    next_or_ot = true;
                    //printf("%d has become true.\n", idx_row);
                }
            }
            else {
                //printf("%d stayed true.\n", idx_row);
            }
            
            //printf("continue? : %d\n", next_or_ot);
            //if (next_or_ot) stride = next_stride;
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
    if (idx_row == 1)
    printall(a, b, c, d, n);
    
    // 2 by 2
    if (idx_row >= (n - closest_power) && idx_row < (closest_power)) {
        /*
        if (idx_row * BLOCK_SIZE >= 84 && idx_row * BLOCK_SIZE < 120) {
            printf("Thread is: %d, Solving 2 by 2: case 1\n Inputs to solver are: b[idx*b*b]: %f, b[idx*b*b+1]: %f, d[idx*b]: %f, b[idx*b*b+2]: %f, b[idx*b*b+3]: %f, d[idx*b + 1]: %f\n", idx_row, b[idx_row * BLOCK_SIZE * BLOCK_SIZE], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 1], d[idx_row * BLOCK_SIZE], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 2], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 3], d[idx_row * BLOCK_SIZE + 1]);
        }
        */
        solve_2by2(b[idx_row * BLOCK_SIZE * BLOCK_SIZE], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 1], d[idx_row * BLOCK_SIZE], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 2], b[idx_row * BLOCK_SIZE * BLOCK_SIZE + 3], d[idx_row * BLOCK_SIZE + 1], &x[idx_row * BLOCK_SIZE], &x[idx_row * BLOCK_SIZE + 1]);
        //xlist[idx_row] = dlist[idx_row]/blist[idx_row];
    }
    else {
        // 4 by 4
        if (idx_row < n/2) {
            // identify A, B, C, D, and K matrix

            //printf("\ncase 3. Idx is: %d, Stride is: %d\n", idx_row, stride);
        
            solve_4by4_x(idx_row, stride, &b[idx_row * BLOCK_SIZE * BLOCK_SIZE], &c[idx_row * BLOCK_SIZE * BLOCK_SIZE], &a[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], &b[(idx_row + stride) * BLOCK_SIZE * BLOCK_SIZE], &d[idx_row * BLOCK_SIZE], &d[(idx_row + stride) * BLOCK_SIZE], &x[idx_row * BLOCK_SIZE]);
        }
        else {
            //printf("\ncase 4. Idx is: %d, Stride is: %d\n", idx_row, stride);

            solve_4by4_y(idx_row, stride, &b[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], &c[(idx_row - stride) * BLOCK_SIZE * BLOCK_SIZE], &a[idx_row * BLOCK_SIZE * BLOCK_SIZE], &b[idx_row * BLOCK_SIZE * BLOCK_SIZE], &d[(idx_row - stride) * BLOCK_SIZE], &d[idx_row * BLOCK_SIZE], &x[idx_row * BLOCK_SIZE]);
        }
    }
}

__host__ 
void generate_block_tridiagonal_system(int n, float *h_a, float *h_b, float *h_c, float *h_d) {
    /*
    float src_a[] = {0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,  -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0};
    float src_b[] = {};2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0
    float src_c[] = {-1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0};
    float src_d[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    */

    /*
    float src_a[] = {0.0, 0.0, 0.0, 0.0, 9.0, 5.0, 5.0, 1.0,  8.0, 5.0, 5.0, -15.0, 91.0, 5.0, 5.0, 2.0};
    float src_b[] = {2.0, 1.0, 3.0, 5.0, 2.0, 9.0, 4.0, 2.0, 2.0, 4.0, 11.0, 2.0, 2.0, -1.0, -11.0, 2.0};
    float src_c[] = {-11.0, 10.0, 10.0, 8.0, 8.0, 10.0, 17.0, 81.0, -71.0, 10.0, 101.0, 99.0, 0.0, 0.0, 0.0, 0.0};
    float src_d[] = {5.0, 100.0, 9.0, -14.0, 5.0, 17.0, 80.0, 7.0};
    */
    

    /*
    float src_a[] = {0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 9.0, 2.0, 1.0, 7.0, 2.0, 5.0};
    float src_b[] = {14.0, 6.0, 1.0, 20.0, 11.0, 1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 9.0};
    float src_c[] = {4.0, 4.0, 8.0, 5.0, 2.0, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0};
    float src_d[] = {13.0, 9.0, 8.0, 8.0, 5.0, 2.0};
    */
    
    float src_a[] = {0.000000, 0.000000, 0.000000, 0.000000, 75.000000, 88.000000, 62.000000, 4.000000, 78.000000, 90.000000, 35.000000, 84.000000, 75.000000, 45.000000, 48.000000, 26.000000, 84.000000, 21.000000, 97.000000, 50.000000};
    float src_b[] = {151.000000, 107.000000, 148.000000, 170.000000, 235.000000, 202.000000, 242.000000, 107.000000, 230.000000, 208.000000, 201.000000, 231.000000, 230.000000, 201.000000, 261.000000, 236.000000, 226.000000, 96.000000, 196.000000, 125.000000};
    float src_c[] = {62.000000, 3.000000, 70.000000, 69.000000, 84.000000, 58.000000, 89.000000, 29.000000, 24.000000, 49.000000, 64.000000, 83.000000, 37.000000, 18.000000, 92.000000, 80.000000, 0.000000, 0.000000, 0.000000, 0.000000};
    float src_d[] = {25.000000, 80.000000, 16.000000, 73.000000, 99.000000, 50.000000, 88.000000, 4.000000, 85.000000, 70.000000};

    memcpy(h_a, src_a, sizeof(src_a));
    memcpy(h_b, src_b, sizeof(src_b));
    memcpy(h_c, src_c, sizeof(src_c));
    memcpy(h_d, src_d, sizeof(src_d));
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
void print_to_fp(int n, float diag[], FILE *fp) {
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%f, ", diag[i]);
    }
}

__host__
int main() {

    // Open a CSV file to save the execution times
    char file_path[512];
    int sizes[] = {128};
    time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    sprintf(file_path, "execution_times_%d_%s.csv", sizes[0], asctime (timeinfo));
    FILE *fp = fopen("execution_times.csv", "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file for writing.\n");
        return -1;
    }
    fprintf(fp, "System Size,Execution Time (ms)\n"); // Write the CSV header

    //int sizes[] = {2, 5, 10, 100, 500, 750, 1000, 2000, 5000}; // Sizes of n.
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);  // Number of different system sizes

    int num_systems = 0;
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

            printf("Solution vector x:\n");
            fprintf(fp, "Solution vector for system %d:\n", i);
            for (int i = 0; i < n*BLOCK_SIZE; i++) {
                printf("%f, ", h_x[i]);
                fprintf(fp, "%f, ", h_x[i]);
            }

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

            printf("number of blocks %d\n", blocks);
        }

        //printf("Average runtime: %f\n", totaltime/num_systems);
        //fprintf(fp,"%d,%f\n", n,totaltime/num_systems);
    }

    fclose(fp);
    return 0;
}
