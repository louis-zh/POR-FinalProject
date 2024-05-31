#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10 // matrix size

__global__ void list_print(int nmax, float * in) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    printf("Thread %i shows %f \n", i, in[i]);
}


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


__global__ void Solve_Kernel(
    float * alist, float * blist, float * clist, float * dlist, float * xlist, int iter_max, int DMax) {

    int idx_row = blockIdx.x*blockDim.x + threadIdx.x;
    int row_max = DMax - 1;

    int stride = 1;
    int next_stride = stride;

    float a1, b1, c1, d1;
    float k01, k21, c01, a21, d01, d21;

    bool next_or_ot = true;

    float i_end = log2f(N);


    for (int i = 0; i < i_end - 1; ++i) {
        printf("iteration : %d, thread is : %d, next or not is: %d\n", i, idx_row, next_or_ot);
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

        // print
        __syncthreads();

        alist[idx_row] = a1;
        blist[idx_row] = b1;
        clist[idx_row] = c1;
        dlist[idx_row] = d1;
    }

    // Solve!

    // When N is odd, we have one row left out
    if ((N & 1) && idx_row == int (N/2))   {
        xlist[idx_row] = dlist[idx_row]/blist[idx_row];
    }
    else if ((N & 2) && (idx_row == int (N/2) || idx_row == int (N/2) - 1)) {
        xlist[idx_row] = dlist[idx_row]/blist[idx_row];
    }
    // When N is even, all reductions will leave us with 2 by 2 matrices
    else {
        if (idx_row < N/2) {
            solve_2by2_x(blist[idx_row], clist[idx_row], dlist[idx_row], alist[idx_row + stride], blist[idx_row + stride], dlist[idx_row + stride] , &xlist[idx_row]);
        }
        else {
            solve_2by2_y(blist[idx_row - stride], clist[idx_row - stride], dlist[idx_row - stride], alist[idx_row], blist[idx_row], dlist[idx_row] , &xlist[idx_row]);
        }
    }
    

    // When N is odd, we have one row left out
    


}

__host__
int main() {

    // Host arrays

    /*
    float h_a[] = {0, -1, -1, -1, -1, -1};
    float h_b[] = {2, 2, 2, 1, 3, 5};
    float h_c[] = {-1, -1, -1, -1, -1, 0};
    float h_d[] = {0, 0, 1, 0, 1, 5};
    float h_x[N];
    */

    // 7
    /*
    float h_a[] = {0, -1, -1, -1, -1, -1, -2};
    float h_b[] = {2, 2, 2, 1, 3, 5, 7};
    float h_c[] = {-1, -1, -1, -1, -1, -1, 0};
    float h_d[] = {0, 0, 1, 0, 1, 5, 2};
    float h_x[N];
    */

    // 8 
    /*
    float h_a[] = {0, -1, -1, -1, -1, -1, -1, -1};
    float h_b[] = {2, 2, 2, 1, 3, 5, 7, 3};
    float h_c[] = {-1, -1, -1, -1, -1, -1, -1, 0};
    float h_d[] = {0, 0, 1, 0, 1, 5, 2, 4};
    float h_x[N];
    */

    // 9
    /*
    float h_a[] = {0, -1, -1, -1, -1, -1, -1, -1, -1};
    float h_b[] = {2, 2, 2, 1, 3, 5, 7, 3, 4};
    float h_c[] = {-1, -1, -1, -1, -1, -1, -1, -1, 0};
    float h_d[] = {0, 0, 1, 0, 1, 5, 2, 4, 9};
    float h_x[N];
    */

    float h_a[] = {0, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    float h_b[] = {2, 2, 2, 1, 3, 5, 7, 3, 4, 6};
    float h_c[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0};
    float h_d[] = {0, 0, 1, 0, 1, 5, 2, 4, 9, 5};
    float h_x[N];

    
    
    
    /*
    float h_a[] = {0, -0.5, -1};
    float h_b[] = {1.5, 0.5, 2.0};
    float h_c[] = {-0.5, -1, 0};
    float h_d[] = {0, 1, 1};
    float h_x[N];
    */
    
    
    /*
    float h_a[] = {0, -1, -1};
    float h_b[] = {2, 2, 2};
    float h_c[] = {-1, -1, 0};
    float h_d[] = {0, 0, 1};
    float h_x[N];
    */

    /*

     // Output the results
    printf("\na vector x:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_a[i]);
    }
     // Output the results
    printf("\nb vector x:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_b[i]);
    }
     // Output the results
    printf("\nc vector x:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_c[i]);
    }
     // Output the results
    printf("\nd vector x:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_d[i]);
    }

    */


    // Device arrays
    float *d_a, *d_b, *d_c, *d_d, *d_x;

    // Allocate memory on the device
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMalloc(&d_d, N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define kernel launch configuration
    int threadsPerBlock = N;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    Solve_Kernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, d_d, d_x, 10, N);

    // Copy results back to host
    cudaMemcpy(h_a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d, d_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n\nafter computation: \n\n");

     // Output the results
    printf("\na vector x:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_a[i]);
    }
     // Output the results
    printf("\nb vector x:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_b[i]);
    }
     // Output the results
    printf("\nc vector x:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_c[i]);
    }
     // Output the results
    printf("\nd vector x:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_d[i]);
    }

    // Output the results
    printf("\nSolution vector x:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_x[i]);
    }

    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_x);

    return 0;
}