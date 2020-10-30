// #pragma OPENCL EXTENSION cl_khr_fp64: enable

// Keep Eclipse happy
#ifdef __CDT_PARSER__
#define __global
#define __local
#define __private
#define __kernel
#endif
/*
Divide and Conquer:

We divide A and B in NxN blocks

We compute the product for each of the submatrices and store it locally, and accumulate it. 
The local computation basically uses as many threads as there are points in the submatrix

There is not much point (I think) in storing the submatrices locally

If we have 4K, for int, that means 1K elts or a 32x32. 
If the matrix is mWidth*mWidth that means we need 32x32 groups
*/

__kernel void matmultKernel10 (
    __global int *A,
    __global int *B,
    __global int *C,
    const unsigned int mWidth) {

int ii = get_group_id(1);
int jj = get_group_id(0);
int si=get_local_id(1);
int sj=get_local_id(0);
int sbw=get_local_size(0);
int sbh=get_local_size(1);
int C_si_sj=0.0;
for (int kk = 0 ; kk < mWidth/sbh; kk++) {
    for (int sk=0;sk<sbh;sk++) {
        int A_ii_kk=A[ii*mWidth*sbh+kk*sbw+mWidth*si+sk];
        int B_kk_jj=B[kk*mWidth*sbh+jj*sbw+mWidth*sk+sj];
        C_si_sj+=A_ii_kk*B_kk_jj;
    }
}
// Write C[ii][jj] to the global memory
C[ii*mWidth*sbh+jj*sbw+si*mWidth+sj]=C_si_sj;

} // matmultKernel10

// int4 version
__kernel void matmultKernel11 (
    __global int *A,
    __global int *B,
    __global int *C,
    const unsigned int mWidth) {

int ii = get_group_id(1);
int jj = get_group_id(0);
int si=get_local_id(1);
int sj=get_local_id(0);
int sbw=get_local_size(0); // 64
int sbh=get_local_size(1); // 64
int C_si_sj=0.0;
// loop over all subblocks
for (int kk = 0 ; kk < mWidth/sbh; kk++) {
	// loop over subblock rows
    for (int sk=0;sk<sbh;sk++) {
        int A_ii_kk=A[ii*mWidth*sbh+kk*sbw+mWidth*si+sk];
        int B_kk_jj=B[kk*mWidth*sbh+jj*sbw+mWidth*sk+sj];
        C_si_sj+=A_ii_kk*B_kk_jj;
    }
}
// Write C[ii][jj] to the global memory
C[ii*mWidth*sbh+jj*sbw+si*mWidth+sj]=C_si_sj;

} // matmultKernel11
