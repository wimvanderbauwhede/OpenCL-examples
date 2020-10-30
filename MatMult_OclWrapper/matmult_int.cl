// Keep Eclipse happy
#ifdef __CDT_PARSER__
#define __global
#define __local
#define __private
#define __kernel
#endif
/*
 A simple int matrix multiplication on a 1024x1024 matrix
 The first version computes from global memory
 The second tries to use the local memory
 I have 16KB so I can store 4K ints, means I can load 2 rows and 2 cols at once, that is not much
 The third version uses SIMD, but with 2 rows and 2 cols that is rather limited. I can use int2 bit not int2
*/

// baseline
__kernel void matmultKernel1 (
    __global int *mA,
    __global int *mB,
    __global int *mC,
    const unsigned int mWidth) {

// naive means every kernel does one element 
    //unsigned int idx=get_global_id(0);
    //unsigned int x=idx % mWidth;
    //unsigned int y=idx / mWidth;
    unsigned int x=get_global_id(0);
    unsigned int y=get_global_id(1);
    int elt=0.0;
    for (unsigned int i=0;i<mWidth;i++) {    
        elt+=mA[y*mWidth+i]*mB[i*mWidth+x];
    }
    mC[x+mWidth*y]=elt;

}

// This version of the kernel is an intermediate one, to debug the use of vector ints
__kernel void matmultKernel2 (
    __global int *mA,
    __global int *mB,
    __global int *mC,
    const unsigned int mWidth) {

// every kernel does 4 elements 
    int pos_x = get_global_id(0);
    int pos_y = get_global_id(1);

    int i=pos_y*2;
    int j=pos_x*2;
    int elt00=0.0;
    int elt01=0.0;
    int elt10=0.0;
    int elt11=0.0;
    for (unsigned int k=0;k<mWidth;k++) {    
        elt00+=mA[i*mWidth+k]*mB[k*mWidth+j];
        elt01+=mA[i*mWidth+k]*mB[k*mWidth+j+1];
        elt10+=mA[(i+1)*mWidth+k]*mB[k*mWidth+j];
        elt11+=mA[(i+1)*mWidth+k]*mB[k*mWidth+j+1];
    }
    mC[(j+0)+mWidth*(i+0)]=elt00;
    mC[(j+1)+mWidth*(i+0)]=elt01;
    mC[(j+0)+mWidth*(i+1)]=elt10;
    mC[(j+1)+mWidth*(i+1)]=elt11;

}

// Canonical 2x2

// This version of the kernel uses int2
// this means we effectively read 2 elements of each row of A at once and one element of 2 cols of B 
// So we compute 4 results per block, so we need half as many blocks
// This kernel makes the Tesla hang, and the GeForce GT 120 as well!
// But somehow only with USE_HOST_PTR?
__kernel void matmultKernel3 (
    __global int2 *mA,
    __global int2 *mB,
    __global int2 *mC,
    const unsigned int mWidth) {

// every kernel does 4 elements 
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int i=(pos.y<<1);
    int j=(pos.x<<1);

    int2 elt0=(int2)(0.0,0.0);
    int2 elt1=(int2)(0.0,0.0);
    for (unsigned int k=0;k<mWidth;k+=2) {    

        int2 mA0=mA[(i*mWidth+k)/2];
        int2 mA1=mA[((i+1)*mWidth+k)/2];
        int2 mB0=mB[(k*mWidth+j)/2];
        int2 mB1=mB[((k+1)*mWidth+j)/2];

// Somehow, this is wrong?!
        /*
        elt0.x+=mA0.x*mB0.x+mA0.y*mB1.x;
        elt0.y+=mA0.x*mB0.y+mA0.y*mB1.y;
        elt1.x+=mA1.x*mB0.x+mA1.y*mB1.x;
        elt1.y+=mA1.x*mB0.y+mA1.y*mB1.y;
        */

// But this works correctly ...        

        elt0.x+=mA0.x*mB0.x; elt0.x+=mA0.y*mB1.x;
        elt0.y+=mA0.x*mB0.y; elt0.y+=mA0.y*mB1.y;
        elt1.x+=mA1.x*mB0.x; elt1.x+=mA1.y*mB1.x;
        elt1.y+=mA1.x*mB0.y; elt1.y+=mA1.y*mB1.y;

    }
    
    mC[(j+mWidth*(i+0))/2]=elt0;
    mC[(j+mWidth*(i+1))/2]=elt1;

}


// Vectorised 4x4. no caching
__kernel void matmultKernel4 (
    __global int4 *mA,
    __global int4 *mB,
    __global int4 *mC,
    const unsigned int mWidth) {

// every kernel does 4 elements 
 
	int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int i=(pos.y<<2);
    int j4=pos.x;
    int j=(j4<<2);
    
    
        int4 elt0=(int4)(0.0,0.0,0.0,0.0);
        int4 elt1=(int4)(0.0,0.0,0.0,0.0);
        int4 elt2=(int4)(0.0,0.0,0.0,0.0);
        int4 elt3=(int4)(0.0,0.0,0.0,0.0);
        unsigned int mWidth4 = mWidth/4;
        for (unsigned int k=0;k<mWidth;k+=4) {    
        	unsigned int k4=k>>2;
			int4 mA0=mA[(i+0)*mWidth4+k4];
			int4 mA1=mA[(i+1)*mWidth4+k4];
			int4 mA2=mA[(i+2)*mWidth4+k4];
			int4 mA3=mA[(i+3)*mWidth4+k4];
			
			int4 mB0=mB[(k+0)*mWidth4+j4];
			int4 mB1=mB[(k+1)*mWidth4+j4];
			int4 mB2=mB[(k+2)*mWidth4+j4];
			int4 mB3=mB[(k+3)*mWidth4+j4];
 
			elt0.s0+=mA0.s0*mB0.s0; elt0.s0+=mA0.s1*mB1.s0; elt0.s0+=mA0.s2*mB2.s0; elt0.s0+=mA0.s3*mB3.s0;
			elt0.s1+=mA0.s0*mB0.s1; elt0.s1+=mA0.s1*mB1.s1; elt0.s1+=mA0.s2*mB2.s1; elt0.s1+=mA0.s3*mB3.s1;
			elt0.s2+=mA0.s0*mB0.s2; elt0.s2+=mA0.s1*mB1.s2; elt0.s2+=mA0.s2*mB2.s2; elt0.s2+=mA0.s3*mB3.s2;
			elt0.s3+=mA0.s0*mB0.s3; elt0.s3+=mA0.s1*mB1.s3; elt0.s3+=mA0.s2*mB2.s3; elt0.s3+=mA0.s3*mB3.s3;
			
			elt1.s0+=mA1.s0*mB0.s0; elt1.s0+=mA1.s1*mB1.s0; elt1.s0+=mA1.s2*mB2.s0; elt1.s0+=mA1.s3*mB3.s0;
			elt1.s1+=mA1.s0*mB0.s1; elt1.s1+=mA1.s1*mB1.s1; elt1.s1+=mA1.s2*mB2.s1; elt1.s1+=mA1.s3*mB3.s1;
			elt1.s2+=mA1.s0*mB0.s2; elt1.s2+=mA1.s1*mB1.s2; elt1.s2+=mA1.s2*mB2.s2; elt1.s2+=mA1.s3*mB3.s2;
			elt1.s3+=mA1.s0*mB0.s3; elt1.s3+=mA1.s1*mB1.s3; elt1.s3+=mA1.s2*mB2.s3; elt1.s3+=mA1.s3*mB3.s3;

			elt2.s0+=mA2.s0*mB0.s0; elt2.s0+=mA2.s1*mB1.s0; elt2.s0+=mA2.s2*mB2.s0; elt2.s0+=mA2.s3*mB3.s0;
			elt2.s1+=mA2.s0*mB0.s1; elt2.s1+=mA2.s1*mB1.s1; elt2.s1+=mA2.s2*mB2.s1; elt2.s1+=mA2.s3*mB3.s1;
			elt2.s2+=mA2.s0*mB0.s2; elt2.s2+=mA2.s1*mB1.s2; elt2.s2+=mA2.s2*mB2.s2; elt2.s2+=mA2.s3*mB3.s2;
			elt2.s3+=mA2.s0*mB0.s3; elt2.s3+=mA2.s1*mB1.s3; elt2.s3+=mA2.s2*mB2.s3; elt2.s3+=mA2.s3*mB3.s3;

			elt3.s0+=mA3.s0*mB0.s0; elt3.s0+=mA3.s1*mB1.s0; elt3.s0+=mA3.s2*mB2.s0; elt3.s0+=mA3.s3*mB3.s0;
			elt3.s1+=mA3.s0*mB0.s1; elt3.s1+=mA3.s1*mB1.s1; elt3.s1+=mA3.s2*mB2.s1; elt3.s1+=mA3.s3*mB3.s1;
			elt3.s2+=mA3.s0*mB0.s2; elt3.s2+=mA3.s1*mB1.s2; elt3.s2+=mA3.s2*mB2.s2; elt3.s2+=mA3.s3*mB3.s2;
			elt3.s3+=mA3.s0*mB0.s3; elt3.s3+=mA3.s1*mB1.s3; elt3.s3+=mA3.s2*mB2.s3; elt3.s3+=mA3.s3*mB3.s3;
        }        
       	
				mC[(i+0)*mWidth4+j4]=elt0;	
				mC[(i+1)*mWidth4+j4]=elt1;
				mC[(i+2)*mWidth4+j4]=elt2;
				mC[(i+3)*mWidth4+j4]=elt3;
				
}

// This version of the kernel uses int2 and a local cache for the cols
// this means we effectively read 2 elements of each row of A at once and one element of 2 cols of B 
// So we compute 4 results per block, so we need half as many blocks
__kernel void matmultKernel5 (
    __global int2 *mA,
    __global int2 *mB,
    __global int2 *mC,
    const unsigned int mWidth) {

// every kernel does 4 elements 
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int i=(pos.y<<1);
    int j=(pos.x<<1);
    int j2=pos.x;//__local 
    int2 mBcols[1024]; // this is 1024*2*4 or 8K
    // Store 2 cols, .s0 and .s1
    for (unsigned int k=0;k<mWidth;k+=2) {
    	mBcols[k]=mB[(k*mWidth+j)/2];
    	mBcols[k+1]=mB[((k+1)*mWidth+j)/2];
    }
    int2 elt0=(int2)(0.0,0.0);
    int2 elt1=(int2)(0.0,0.0);
    for (unsigned int k=0;k<mWidth;k+=2) {    

        int2 mA0=mA[(i*mWidth+k)/2];
        int2 mA1=mA[((i+1)*mWidth+k)/2];
        
        int2 mB0=mBcols[k+0];
        int2 mB1=mBcols[k+1];

//        int2 mB0=mB[(k*mWidth+j)/2];
//        int2 mB1=mB[((k+1)*mWidth+j)/2];
        
        elt0.x+=mA0.x*mB0.x; elt0.x+=mA0.y*mB1.x;
        elt0.y+=mA0.x*mB0.y; elt0.y+=mA0.y*mB1.y;
        elt1.x+=mA1.x*mB0.x; elt1.x+=mA1.y*mB1.x;
        elt1.y+=mA1.x*mB0.y; elt1.y+=mA1.y*mB1.y;

    }
    
    mC[(j+mWidth*(i+0))/2]=elt0;
    mC[(j+mWidth*(i+1))/2]=elt1;
    
}

// Vectorised 1x4 with use of local memory to cache colums

__kernel void matmultKernel7 (
    __global int4 *mA,
    __global int *mB,
    __global int *mC,
    const unsigned int mWidth) {

// Every kernel does 1x4 elements
// We load 1 column into the local memory once
// Then we compute the 1x4 elements using the local values

	int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int i=pos.y; // row, so mWidth
    int j=pos.x; // col, so mWidth/4
    unsigned int mWidth4 = mWidth/4;

    int mBcols[1024]; // this is 1024*4 or 4K;

    for (unsigned int k=0;k<mWidth;k++) {
    	mBcols[k]=mB[k*mWidth+j];
    }

	int elt0=0.0;
	for (unsigned int k=0;k<mWidth;k+=4) { // read a row from A
		unsigned int k4=k>>2;
		int4 mA0=mA[i*mWidth4+k4];

		int mB0=mBcols[(k+0)];
		int mB1=mBcols[(k+1)];
		int mB2=mBcols[(k+2)];
		int mB3=mBcols[(k+3)];

		elt0+=mA0.s0*mB0; elt0+=mA0.s1*mB1; elt0+=mA0.s2*mB2; elt0+=mA0.s3*mB3;
	}

	mC[i*mWidth+j]=elt0;

}
/*
// The kernel below lead to an error in matmult_int.pl
// ERROR: loadKernel::Kernel() (-45)
// because I used NTH (now nth) which is also used as a macro
//
*/

// We read 1 column per group and cache it. Then we use it for N threads in this group.
// So we have 1024x(1024/64) groups of 64 threads
__kernel void matmultKernel8 (
//    __global int4 *mA,
    __global int *mA,
    __global int *mB,
    __global int *mC,
    const unsigned int mWidth) {

// Every kernel does 1x4 elements
// We load 1 column into the local memory once
// Then we compute the 1x4 elements using the local values

    int j=get_group_id(0); // col, so mWidth
	int tid = get_local_id(0);
	int nth=get_local_size(0);
	// cache a column
//    __local int mBcol[1024]; // this is 1024*4 or 4K and it's too big on iMac and MacBook
    __local int mBcol[512];
	for (unsigned int k=mWidth*tid/nth;k<(tid+1)*mWidth/nth;k+=1) {
		if (k%2==0)
			mBcol[k/2]=mB[k*mWidth+j];
//		mBcol[k/2]=mB[k*mWidth+j];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// compute all elts of C for that column
	for (unsigned int i=tid*mWidth/nth;i<(tid+1)*mWidth/nth;i++) {
		int elt=0.0;
		for (unsigned int k=0;k<mWidth;k++) {
//			elt+=mA[i*mWidth+k]*mBcol[k];
			unsigned int kk=(k+tid+j)%mWidth;
			if (kk%2==0) {
			elt+=mA[i*mWidth+kk]*mBcol[kk/2];
			} else {
			elt+=mA[i*mWidth+kk]*mB[kk*mWidth+j];
			}
		}
		mC[i*mWidth+j]=elt;
	}
}




/*
__kernel void matmultKernel9 ( // for the Tesla
    __global int4 *mA,
    __global int4 *mB,
    __global int4 *mC,
    const unsigned int mWidth) {

// Every kernel does 1x4 elements
// We load 1 column into the local memory once
// Then we compute the 1x4 elements using the local values

    int j=get_group_id(0); // col, so mWidth/4
	int tid = get_local_id(0);
	int NTH=get_local_size(0);
	const unsigned int mWidth4 = mWidth/4;
	// cache 4 columns
    __local int4 mBcol[1024];
	for (unsigned int k=mWidth*tid/NTH;k<(tid+1)*mWidth/NTH;k+=1) {
		mBcol[k]=mB[k*mWidth4+j];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// compute all elts of C for these 4 columns
	// every thread processes mWidth/NTH rows
	for (unsigned int i=tid*mWidth/NTH;i<(tid+1)*mWidth/NTH;i++) {
		int4 elt=(int4)(0.0);
		// loop over the elts in a row
		for (unsigned int kk4=0;kk4<mWidth4;kk4++) {
			unsigned int kk=kk4*4;
			elt.s0+=mA[i*mWidth4+kk4].s0*mBcol[kk].s0;
			elt.s0+=mA[i*mWidth4+kk4].s1*mBcol[kk+1].s0;
			elt.s0+=mA[i*mWidth4+kk4].s2*mBcol[kk+2].s0;
			elt.s0+=mA[i*mWidth4+kk4].s3*mBcol[kk+3].s0;

			elt.s1+=mA[i*mWidth4+kk4].s0*mBcol[kk].s1;
			elt.s1+=mA[i*mWidth4+kk4].s1*mBcol[kk+1].s1;
			elt.s1+=mA[i*mWidth4+kk4].s2*mBcol[kk+2].s1;
			elt.s1+=mA[i*mWidth4+kk4].s3*mBcol[kk+3].s1;

			elt.s2+=mA[i*mWidth4+kk4].s0*mBcol[kk].s2;
			elt.s2+=mA[i*mWidth4+kk4].s1*mBcol[kk+1].s2;
			elt.s2+=mA[i*mWidth4+kk4].s2*mBcol[kk+2].s2;
			elt.s2+=mA[i*mWidth4+kk4].s3*mBcol[kk+3].s2;

			elt.s3+=mA[i*mWidth4+kk4].s0*mBcol[kk].s3;
			elt.s3+=mA[i*mWidth4+kk4].s1*mBcol[kk+1].s3;
			elt.s3+=mA[i*mWidth4+kk4].s2*mBcol[kk+2].s3;
			elt.s3+=mA[i*mWidth4+kk4].s3*mBcol[kk+3].s3;

		}
		mC[i*mWidth4+j]=elt;
	}
}
*/

/*
   in each WG g_id, we loop over part of a row, the index is th_range*th_id+ii, for all k
   in each thread, we loop over part of a column, the col index is wg_range*g_id+jj, for all k
   k loops 0 .. mWidth-1
   so we have A[(th_range*th_id+ii)*mWidth+k]
   and B[ wg_range*g_id+jj + k*mWidth]

// This code returns the WRONG result on Intel CPU, I guess maybe because the number of cores is 12 so the division is incorrect
// On the MIC it is also *very* slow!
*/
__kernel void matmultKernel11 (
    __global uint *mA,
    __global uint *mB,
    __global uint *mC,
    unsigned int mWidth) {

    uint wg_id=get_group_id(0);
    uint nunits = get_num_groups(0);
    uint wg_range = mWidth/nunits;
/*
        uint th_id = get_local_id(0);
        uint n_threads=get_local_size(0);
        uint th_range = mWidth/n_threads;
*/
        //uint th_range = wg_range/n_threads;

//    for (uint jj = wg_id*wg_range+th_id*th_range;jj<wg_id*wg_range+(th_id+1)*th_range;jj++) { // loop over part of a row
  //      for (unsigned int ii=0;ii<mWidth;ii++) {

    for (uint jj = wg_id*wg_range;jj<(wg_id+1)*wg_range;jj++) { // loop over part of a row
/*
	uint th_id = get_local_id(0);
	uint n_threads=get_local_size(0);
	uint th_range = wg_range/n_threads;
    for (uint jj = wg_id*wg_range+th_id*th_range;jj<wg_id*wg_range+(th_id+1)*th_range;jj++) { // loop over part of a row
*/
        for (unsigned int ii=0;ii<mWidth;ii++) {
            uint elt=0;
            for (unsigned int k=0;k<mWidth;k++) {
                elt+=mB[jj+mWidth*k]*mA[k+ii*mWidth];
            }
            mC[ii*mWidth+jj]=elt;
        }
    }
  //      barrier(CLK_GLOBAL_MEM_FENCE);
  }

