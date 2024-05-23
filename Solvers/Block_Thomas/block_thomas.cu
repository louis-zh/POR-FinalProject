#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

__device__
void matrixMultiply(double *A, double *B, double *C, int m, int k, int n) {
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
void matrixSubtract(double *A, double *B, double *C, int m, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            C[row * n + col] = A[row * n + col] - B[row * n + col];
        }
    }
}

__device__
void invertMatrix(double *mat, double *invMat, int n) {
    // This simple inversion is appropriate only for 2x2 matrices
    double a = mat[0], b = mat[1], c = mat[2], d = mat[3];
    double det = a * d - b * c;
    if (det == 0) {
        printf("Matrix is singular and cannot be inverted.\n");
        return;
    }
    double invDet = 1.0 / det;
    invMat[0] =  d * invDet;
    invMat[1] = -b * invDet;
    invMat[2] = -c * invDet;
    invMat[3] =  a * invDet;
}

__global__
void blockTridiagonalSolver(double *a, double *b, double *c, double *d, double *x, int N, int blockSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx != 0) return;  // Ensure only one thread does the work

    int matSize = blockSize * blockSize;
    int vecSize = blockSize;  // Simplified vector size handling
    double *cp = (double *)malloc((N-1) * matSize * sizeof(double));
    double *dp = (double *)malloc(N * vecSize * sizeof(double));
    double *A_inv = (double *)malloc(matSize * sizeof(double));
    double *temp = (double *)malloc(matSize * sizeof(double));

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
    memcpy(x + (N-1) * vecSize, dp + (N-1) * vecSize, vecSize * sizeof(double));
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
int main() {
    const int N = 3;
    const int blockSize = 2;

    double h_X[N * blockSize];
    double h_B[N * blockSize * blockSize] = {4, 1, 1, 3, 5, 2, 2, 4, 6, 1, 1, 5};
    double h_C[(N-1) * blockSize * blockSize] = {-1, 0, 0, -1, -1, 0, 0, -1};
    double h_A[(N-1) * blockSize * blockSize] = {-2, 0, 0, -2, -2, 0, 0, -2};
    double h_D[N * blockSize] = {1, 2, 3, 4, 5, 6};

    // Device memory pointers
    double *d_A, *d_B, *d_C, *d_D, *d_X;

    // Allocate memory on the device
    cudaMalloc(&d_A, sizeof(h_A));
    cudaMalloc(&d_B, sizeof(h_B));
    cudaMalloc(&d_C, sizeof(h_C));
    cudaMalloc(&d_D, sizeof(h_D));
    cudaMalloc(&d_X, N * blockSize * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(h_C), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, sizeof(h_D), cudaMemcpyHostToDevice);

    // Launch kernel with a single thread
    blockTridiagonalSolver<<<1, 1>>>(d_A, d_B, d_C, d_D, d_X, N, blockSize);

    // Copy result back to host
    cudaMemcpy(h_X, d_X, N * blockSize * sizeof(double), cudaMemcpyDeviceToHost);

    // Print solution
    printf("Solution X:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < blockSize; j++) {
            printf("%f ", h_X[i * blockSize + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_X);

    return 0;
}
