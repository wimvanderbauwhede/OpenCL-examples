// Keep Eclipse happy
#ifdef __CDT_PARSER__
#define __global
#define __local
#define __private
#define __kernel
#endif
/*
 A simple float matrix multiplication on a 1024x1024 matrix
 The first version computes from global memory
 The second tries to use the local memory
 I have 16KB so I can store 4K floats, means I can load 2 rows and 2 cols at once, that is not much
 The third version uses SIMD, but with 2 rows and 2 cols that is rather limited. I can use float2 bit not float2
*/

// baseline
__kernel void matmultKernel1 (
    __global float *mA,
    __global float *mB,
    __global float *mC,
    const unsigned int mWidth) {

// naive means every kernel does one element 
    //unsigned int idx=get_global_id(0);
    //unsigned int x=idx % mWidth;
    //unsigned int y=idx / mWidth;
    unsigned int x=get_global_id(0);
    unsigned int y=get_global_id(1);
    float elt=0.0;
    for (unsigned int i=0;i<mWidth;i++) {    
        elt+=mA[y*mWidth+i]*mB[i*mWidth+x];
    }
    mC[x+mWidth*y]=elt;

}

// This version of the kernel is an intermediate one, to debug the use of vector floats
__kernel void matmultKernel2 (
    __global float *mA,
    __global float *mB,
    __global float *mC,
    const unsigned int mWidth) {

// every kernel does 4 elements 
    int pos_x = get_global_id(0);
    int pos_y = get_global_id(1);

    int i=pos_y*2;
    int j=pos_x*2;
    float elt00=0.0;
    float elt01=0.0;
    float elt10=0.0;
    float elt11=0.0;
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

/*

for (uint pos_y = 0; pos_y <mWidth/2; pos_y ++) {
    int i=pos_y*2;
    for (uint pos_x = 0; pos_x<mWidth/2; pos_x++) {
        int j=pos_x*2;
        float elt[2][2]={{0.0,0.0},{0.0,0.0}};
        float elt[0][0]=0.0;
        float elt[0][1]=0.0;
        float elt[1][0]=0.0;
        float elt[1][1]=0.0;
        for (unsigned int k=0;k<mWidth;k++) {    
            elt[0][0]+=mA[i][k]*mB[k][j];
            elt[0][1]+=mA[i][k]*mB[k][j+1];
            elt[1][0]+=mA[i+1][k]*mB[k][j];
            elt[1][1]+=mA[i+1][k]*mB[k][j+1];
        }
        mC[i][j]=elt00;
        mC[i][j+1]=elt01;
        mC[i+1][j]=elt10;
        mC[i+1][j+1]=elt11;
    }
    }
*/

// This version of the kernel uses float2
// this means we effectively read 2 elements of each row of A at once and one element of 2 cols of B 
// So we compute 4 results per block, so we need half as many blocks
// This kernel makes the Tesla hang, and the GeForce GT 120 as well!
// But somehow only with USE_HOST_PTR?
__kernel void matmultKernel3 (
    __global float2 *mA,
    __global float2 *mB,
    __global float2 *mC,
    const unsigned int mWidth) {

// every kernel does 4 elements 
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int i=(pos.y<<1);
    int j=(pos.x<<1);

    float2 elt0=(float2)(0.0,0.0);
    float2 elt1=(float2)(0.0,0.0);
    for (unsigned int k=0;k<mWidth;k+=2) {    

        float2 mA0=mA[(i*mWidth+k)/2];
        float2 mA1=mA[((i+1)*mWidth+k)/2];
        float2 mB0=mB[(k*mWidth+j)/2];
        float2 mB1=mB[((k+1)*mWidth+j)/2];

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

// Canonical 4x4

/*
for (uint pos_y = 0; pos_y <mWidth/4; pos_y ++) {
    int i=pos_y*4;
    for (uint pos_x = 0; pos_x<mWidth/4; pos_x++) {
        int j=pos_x*4;
        float elt[4][4]={{0.0,0.0,0.0,0.0},{0.0,0.0,0.0,0.0},{0.0,0.0,0.0,0.0},{0.0,0.0,0.0,0.0}};
        for (unsigned int k=0;k<mWidth;k++) {    
            elt[0][0]+=mA[i+0][k]*mB[k][j+0];
            elt[0][1]+=mA[i+0][k]*mB[k][j+1];
            elt[0][2]+=mA[i+0][k]*mB[k][j+2];
            elt[0][3]+=mA[i+0][k]*mB[k][j+3];                       
            
            elt[1][0]+=mA[i+1][k]*mB[k][j+0];
            elt[1][1]+=mA[i+1][k]*mB[k][j+1];
            elt[1][2]+=mA[i+1][k]*mB[k][j+2];
            elt[1][3]+=mA[i+1][k]*mB[k][j+3];
            
            elt[2][0]+=mA[i+2][k]*mB[k][j+0];
            elt[2][1]+=mA[i+2][k]*mB[k][j+1];
            elt[2][2]+=mA[i+2][k]*mB[k][j+2];
            elt[2][3]+=mA[i+2][k]*mB[k][j+3];                       
            
            elt[3][0]+=mA[i+3][k]*mB[k][j+0];
            elt[3][1]+=mA[i+3][k]*mB[k][j+1];
            elt[3][2]+=mA[i+3][k]*mB[k][j+2];
            elt[3][3]+=mA[i+3][k]*mB[k][j+3];
        }
        for (int k=0;k<4;k++) {
			for (int l=0;l<4;l++) {
				mC[i+k][j+l]=elt[k][l];
			}
		}
    }
}
*/

// Vectorised 4x4. no caching
__kernel void matmultKernel4 (
    __global float4 *mA,
    __global float4 *mB,
    __global float4 *mC,
    const unsigned int mWidth) {

// every kernel does 4 elements 
 
	int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int i=(pos.y<<2);
    int j4=pos.x;
    int j=(j4<<2);
    
    
        float4 elt0=(float4)(0.0,0.0,0.0,0.0);
        float4 elt1=(float4)(0.0,0.0,0.0,0.0);
        float4 elt2=(float4)(0.0,0.0,0.0,0.0);
        float4 elt3=(float4)(0.0,0.0,0.0,0.0);
        unsigned int mWidth4 = mWidth/4;
        for (unsigned int k=0;k<mWidth;k+=4) {    
        	unsigned int k4=k>>2;
			float4 mA0=mA[(i+0)*mWidth4+k4];
			float4 mA1=mA[(i+1)*mWidth4+k4];
			float4 mA2=mA[(i+2)*mWidth4+k4];
			float4 mA3=mA[(i+3)*mWidth4+k4];
			
			float4 mB0=mB[(k+0)*mWidth4+j4];
			float4 mB1=mB[(k+1)*mWidth4+j4];
			float4 mB2=mB[(k+2)*mWidth4+j4];
			float4 mB3=mB[(k+3)*mWidth4+j4];
// from k+=1 to k+=4, unroll the loop      
//            elt[0].s0+=mA[((i+0)*mWidth+k)/4].s0*mB[(k*mWidth+j)/4].s0;
//            elt[0].s0+=mA[((i+0)*mWidth+k)/4].s1*mB[((k+1)*mWidth+j)/4].s0;
//            elt[0].s0+=mA[((i+0)*mWidth+k)/4].s2*mB[((k+2)*mWidth+j)/4].s0;
//            elt[0].s0+=mA[((i+0)*mWidth+k)/4].s3*mB[((k+3)*mWidth+j)/4].s0;
 
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

// This version of the kernel uses float2 and a local cache for the cols
// this means we effectively read 2 elements of each row of A at once and one element of 2 cols of B 
// So we compute 4 results per block, so we need half as many blocks
__kernel void matmultKernel5 (
    __global float2 *mA,
    __global float2 *mB,
    __global float2 *mC,
    const unsigned int mWidth) {

// every kernel does 4 elements 
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int i=(pos.y<<1);
    int j=(pos.x<<1);
    int j2=pos.x;//__local 
    float2 mBcols[1024]; // this is 1024*2*4 or 8K
    // Store 2 cols, .s0 and .s1
    for (unsigned int k=0;k<mWidth;k+=2) {
    	mBcols[k]=mB[(k*mWidth+j)/2];
    	mBcols[k+1]=mB[((k+1)*mWidth+j)/2];
    }
    float2 elt0=(float2)(0.0,0.0);
    float2 elt1=(float2)(0.0,0.0);
    for (unsigned int k=0;k<mWidth;k+=2) {    

        float2 mA0=mA[(i*mWidth+k)/2];
        float2 mA1=mA[((i+1)*mWidth+k)/2];
        
        float2 mB0=mBcols[k+0];
        float2 mB1=mBcols[k+1];

//        float2 mB0=mB[(k*mWidth+j)/2];
//        float2 mB1=mB[((k+1)*mWidth+j)/2];
        
        elt0.x+=mA0.x*mB0.x; elt0.x+=mA0.y*mB1.x;
        elt0.y+=mA0.x*mB0.y; elt0.y+=mA0.y*mB1.y;
        elt1.x+=mA1.x*mB0.x; elt1.x+=mA1.y*mB1.x;
        elt1.y+=mA1.x*mB0.y; elt1.y+=mA1.y*mB1.y;

    }
    
    mC[(j+mWidth*(i+0))/2]=elt0;
    mC[(j+mWidth*(i+1))/2]=elt1;
    
}
/*
// Vectorised 2x4 with use of local memory to cache colums
// BROKEN!
__kernel void matmultKernel6 (
    __global float4 *mA,
    __global float4 *mB,
    __global float4 *mC,
    const unsigned int mWidth) {

// Every kernel does 4x4 elements using 4 threads so 4 elements per thread so mWidth*mWidth/4 threads in total
// every work group groups 4 threads
// We load the 2 columns (1 column float2) into the local memory once
// Then we compute the 2x4 elements using the local values
	
	int2 pos = (int2)(get_group_id(0), get_group_id(1));
	int tid = get_local_id(0);
    int i=pos.y<<2;
    int j2=pos.x;
    unsigned int mWidth4 = mWidth/4;
    unsigned int mWidth2 = mWidth/2;
    unsigned int mWidthP = mWidth/4;
    __local float4 elt0s=(float4)(0.0);
    __local float4 elt1s=(float4)(0.0);
    __local float4 elt2s=(float4)(0.0);
    __local float4 elt3s=(float4)(0.0);

    // NOTE: with cl::NullRange it should not be __local
    // With cl::NDRange(1,1), __local should be OK but in practice it crashes, likely because the GPU is driving the screen as well
    float2 mBcols[256]; // this is 1024*2*4 or 8K, and it doesn't fit without __local :-(
    // We have 1/4 of the column for each thread
    for (unsigned int k=tid*mWidthP;k<(tid+1)*mWidthP;k+=4) { // we read 2 columns by 4
    	mBcols[k+0]=mB[(k+0)*mWidth2+j2];
    	mBcols[k+1]=mB[(k+1)*mWidth2+j2];
    	mBcols[k+2]=mB[(k+2)*mWidth2+j2];
    	mBcols[k+3]=mB[(k+3)*mWidth2+j2];
    }

	float2 elt0=(float2)(0.0);
	float2 elt1=(float2)(0.0);
	float2 elt2=(float2)(0.0);
	float2 elt3=(float2)(0.0);

	for (unsigned int k=tid*mWidthP;k<(tid+1)*mWidthP;k+=4) { // loop over all elts in the column, by 4
		unsigned int k4=k>>2;
		float4 mA0=mA[(i+0)*mWidth4+k4];
		float4 mA1=mA[(i+1)*mWidth4+k4];
		float4 mA2=mA[(i+2)*mWidth4+k4];
		float4 mA3=mA[(i+3)*mWidth4+k4];
		
		float2 mB0=mBcols[k+0];			
		float2 mB1=mBcols[k+1];
		float2 mB2=mBcols[k+2];
		float2 mB3=mBcols[k+3];
	
		elt0.s0+=mA0.s0*mB0.s0; elt0.s0+=mA0.s1*mB1.s0; elt0.s0+=mA0.s2*mB2.s0; elt0.s0+=mA0.s3*mB3.s0;
		elt0.s1+=mA0.s0*mB0.s1; elt0.s1+=mA0.s1*mB1.s1; elt0.s1+=mA0.s2*mB2.s1; elt0.s1+=mA0.s3*mB3.s1;
		
		elt1.s0+=mA1.s0*mB0.s0; elt1.s0+=mA1.s1*mB1.s0; elt1.s0+=mA1.s2*mB2.s0; elt1.s0+=mA1.s3*mB3.s0;
		elt1.s1+=mA1.s0*mB0.s1; elt1.s1+=mA1.s1*mB1.s1; elt1.s1+=mA1.s2*mB2.s1; elt1.s1+=mA1.s3*mB3.s1;
	
		elt2.s0+=mA2.s0*mB0.s0; elt2.s0+=mA2.s1*mB1.s0; elt2.s0+=mA2.s2*mB2.s0; elt2.s0+=mA2.s3*mB3.s0;
		elt2.s1+=mA2.s0*mB0.s1; elt2.s1+=mA2.s1*mB1.s1; elt2.s1+=mA2.s2*mB2.s1; elt2.s1+=mA2.s3*mB3.s1;
	
		elt3.s0+=mA3.s0*mB0.s0; elt3.s0+=mA3.s1*mB1.s0; elt3.s0+=mA3.s2*mB2.s0; elt3.s0+=mA3.s3*mB3.s0;
		elt3.s1+=mA3.s0*mB0.s1; elt3.s1+=mA3.s1*mB1.s1; elt3.s1+=mA3.s2*mB2.s1; elt3.s1+=mA3.s3*mB3.s1;
	}        

	mC[(i+0)*mWidth2+j2]=elt0s;
	mC[(i+1)*mWidth2+j2]=elt1s;
	mC[(i+2)*mWidth2+j2]=elt2s;
	mC[(i+3)*mWidth2+j2]=elt3s;
			
}

// Vectorised 2x4 with use of local memory to cache columns
// and using 16 threads
// just to see
// BROKEN!
__kernel void matmultKernel18b (
    __global float4 *mA,
    __global float2 *mB,
    __global float2 *mC,
    const unsigned int mWidth) {

// Every kernel does 4x4 elements 
// We load the 4 columns (1 column float4) into the local memory once
// Then we compute the 4x4 elements using the local values
	
	int2 pos = (int2)(get_global_id(0), get_global_id(1));	
	int2 lpos = (int2)(get_local_id(0),get_local_id(1)); // 4x4=16
	int tid = lpos.x+4*lpos.y;
    int i=(pos.y<<2);
    int j2=pos.x;
    unsigned int mWidth4 = mWidth/4;
    unsigned int mWidth2 = mWidth/2;
    __local float2 mBcols[1024]; // this is 1024*2*4 or 8K, and it doesn't fit without __local :-(

    // We can either make them local like this
	float2 elt0[NTH]; // so 16x2x4=128 bytes
	float2 elt1[NTH];
	float2 elt2[NTH];
	float2 elt3[NTH];

    // Split this into 16 for each thread col; it could easily be more than 4
	// and run one computation per row
	// This is OK: all threads are unconnected
     for (unsigned int k=tid*mWidth/NTH;k<(tid+1)*mWidth/NTH;k+=4) {
    	mBcols[k+0]=mB[(k+0)*mWidth2+j2];
    	mBcols[k+1]=mB[(k+1)*mWidth2+j2];
    	mBcols[k+2]=mB[(k+2)*mWidth2+j2];
    	mBcols[k+3]=mB[(k+3)*mWidth2+j2];
    }

	elt0[tid]=(float2)(0.0);
	elt1[tid]=(float2)(0.0);
	elt2[tid]=(float2)(0.0);
	elt3[tid]=(float2)(0.0);
	
    for (unsigned int k=tid*mWidth/NTH;k<(tid+1)*mWidth/NTH;k+=4) {
		unsigned int k4=k>>2;
		float4 mA0=mA[(i+0)*mWidth4+k4];
		float4 mA1=mA[(i+1)*mWidth4+k4];
		float4 mA2=mA[(i+2)*mWidth4+k4];
		float4 mA3=mA[(i+3)*mWidth4+k4];
		
		float2 mB0=mBcols[k+0];			
		float2 mB1=mBcols[k+1];
		float2 mB2=mBcols[k+2];
		float2 mB3=mBcols[k+3];
	
		elt0[tid].s0+=mA0.s0*mB0.s0; elt0[tid].s0+=mA0.s1*mB1.s0; elt0[tid].s0+=mA0.s2*mB2.s0; elt0[tid].s0+=mA0.s3*mB3.s0;
		elt0[tid].s1+=mA0.s0*mB0.s1; elt0[tid].s1+=mA0.s1*mB1.s1; elt0[tid].s1+=mA0.s2*mB2.s1; elt0[tid].s1+=mA0.s3*mB3.s1;
		
		elt1[tid].s0+=mA1.s0*mB0.s0; elt1[tid].s0+=mA1.s1*mB1.s0; elt1[tid].s0+=mA1.s2*mB2.s0; elt1[tid].s0+=mA1.s3*mB3.s0;
		elt1[tid].s1+=mA1.s0*mB0.s1; elt1[tid].s1+=mA1.s1*mB1.s1; elt1[tid].s1+=mA1.s2*mB2.s1; elt1[tid].s1+=mA1.s3*mB3.s1;
	
		elt2[tid].s0+=mA2.s0*mB0.s0; elt2[tid].s0+=mA2.s1*mB1.s0; elt2[tid].s0+=mA2.s2*mB2.s0; elt2[tid].s0+=mA2.s3*mB3.s0;
		elt2[tid].s1+=mA2.s0*mB0.s1; elt2[tid].s1+=mA2.s1*mB1.s1; elt2[tid].s1+=mA2.s2*mB2.s1; elt2[tid].s1+=mA2.s3*mB3.s1;
	
		elt3[tid].s0+=mA3.s0*mB0.s0; elt3[tid].s0+=mA3.s1*mB1.s0; elt3[tid].s0+=mA3.s2*mB2.s0; elt3[tid].s0+=mA3.s3*mB3.s0;
		elt3[tid].s1+=mA3.s0*mB0.s1; elt3[tid].s1+=mA3.s1*mB1.s1; elt3[tid].s1+=mA3.s2*mB2.s1; elt3[tid].s1+=mA3.s3*mB3.s1;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	for (unsigned int ii=0;ii<NTH;ii++) {
		if (ii==tid) {
		mC[(i+0)*mWidth2+j2].s0=elt0[tid].s0;
		mC[(i+1)*mWidth2+j2].s0=elt1[tid].s0;
		mC[(i+2)*mWidth2+j2].s0=elt2[tid].s0;
		mC[(i+3)*mWidth2+j2].s0=elt3[tid].s0;
		mC[(i+0)*mWidth2+j2].s1=elt0[tid].s1;
		mC[(i+1)*mWidth2+j2].s1=elt1[tid].s1;
		mC[(i+2)*mWidth2+j2].s1=elt2[tid].s1;
		mC[(i+3)*mWidth2+j2].s1=elt3[tid].s1;
		}
	}

} // matmult8b
*/


// Vectorised 1x4 with use of local memory to cache colums

__kernel void matmultKernel7 (
    __global float4 *mA,
    __global float *mB,
    __global float *mC,
    const unsigned int mWidth) {

// Every kernel does 1x4 elements
// We load 1 column into the local memory once
// Then we compute the 1x4 elements using the local values

	int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int i=pos.y; // row, so mWidth
    int j=pos.x; // col, so mWidth/4
    unsigned int mWidth4 = mWidth/4;

    float mBcols[1024]; // this is 1024*4 or 4K;

    for (unsigned int k=0;k<mWidth;k++) {
    	mBcols[k]=mB[k*mWidth+j];
    }

	float elt0=0.0;
	for (unsigned int k=0;k<mWidth;k+=4) { // read a row from A
		unsigned int k4=k>>2;
		float4 mA0=mA[i*mWidth4+k4];

		float mB0=mBcols[(k+0)];
		float mB1=mBcols[(k+1)];
		float mB2=mBcols[(k+2)];
		float mB3=mBcols[(k+3)];

		elt0+=mA0.s0*mB0; elt0+=mA0.s1*mB1; elt0+=mA0.s2*mB2; elt0+=mA0.s3*mB3;
	}

	mC[i*mWidth+j]=elt0;

}

// We read 1 column per group and cache it. Then we use it for N threads in this group.
// So we have 1024x(1024/64) groups of 64 threads
__kernel void matmultKernel8 (
//    __global float4 *mA,
    __global float *mA,
    __global float *mB,
    __global float *mC,
    const unsigned int mWidth) {

// Every kernel does 1x4 elements
// We load 1 column into the local memory once
// Then we compute the 1x4 elements using the local values

    int j=get_group_id(0); // col, so mWidth
	int tid = get_local_id(0);
	int NTH=get_local_size(0);
	// cache a column
//    __local float mBcol[1024]; // this is 1024*4 or 4K and it's too big on iMac and MacBook
    __local float mBcol[512];
	for (unsigned int k=mWidth*tid/NTH;k<(tid+1)*mWidth/NTH;k+=1) {
		if (k%2==0)
			mBcol[k/2]=mB[k*mWidth+j];
//		mBcol[k/2]=mB[k*mWidth+j];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// compute all elts of C for that column
	for (unsigned int i=tid*mWidth/NTH;i<(tid+1)*mWidth/NTH;i++) {
		float elt=0.0;
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

__kernel void matmultKernel9 ( // for the Tesla
    __global float4 *mA,
    __global float4 *mB,
    __global float4 *mC,
    const unsigned int mWidth) {

// Every kernel does 1x4 elements
// We load 1 column into the local memory once
// Then we compute the 1x4 elements using the local values

    int j=get_group_id(0); // col, so mWidth/4
	int tid = get_local_id(0);
	int NTH=get_local_size(0);
	const unsigned int mWidth4 = mWidth/4;
	// cache 4 columns
    __local float4 mBcol[1024];
	for (unsigned int k=mWidth*tid/NTH;k<(tid+1)*mWidth/NTH;k+=1) {
		mBcol[k]=mB[k*mWidth4+j];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// compute all elts of C for these 4 columns
	// every thread processes mWidth/NTH rows
	for (unsigned int i=tid*mWidth/NTH;i<(tid+1)*mWidth/NTH;i++) {
		float4 elt=(float4)(0.0);
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



/*
// Vectorised 1x8 with use of local memory to cache colums
// This is slower than the 1x4!
__kernel void matmultKernel7b (
    __global float8 *mA,
    __global float *mB,
    __global float *mC,
    const unsigned int mWidth) {

// Every kernel does 1x4 elements
// We load 1 column into the local memory once
// Then we compute the 1x4 elements using the local values
	
	int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int i=pos.y; // row, so mWidth
    int j=pos.x; // col, so mWidth/4
    unsigned int mWidth8 = mWidth/8;
    float mBcols[1024]; // this is 1024*4 or 4K;
    for (unsigned int k=0;k<mWidth;k++) {
    	mBcols[k]=mB[k*mWidth+j];
    }
        float elt0=0.0;

        for (unsigned int k=0;k<mWidth;k+=8) { // read a row from A
        	unsigned int k8=k>>3;
			float8 mA0=mA[i*mWidth8+k8];

			float mB0=mBcols[(k+0)];
			float mB1=mBcols[(k+1)];
			float mB2=mBcols[(k+2)];
			float mB3=mBcols[(k+3)];
			float mB4=mBcols[(k+4)];
			float mB5=mBcols[(k+5)];
			float mB6=mBcols[(k+6)];
			float mB7=mBcols[(k+7)];
 
			elt0+=mA0.s0*mB0; elt0+=mA0.s1*mB1; elt0+=mA0.s2*mB2; elt0+=mA0.s3*mB3;
			elt0+=mA0.s4*mB4; elt0+=mA0.s5*mB5; elt0+=mA0.s6*mB6; elt0+=mA0.s7*mB7;
        }        
       	
        mC[i*mWidth+j]=elt0;
				
}

*/

// This is the kernel from the AMD SDK samples. 
// Interestingly, it produces the wrong results ...
#define TILEX 4
#define TILEX_SHIFT 2
#define TILEY 4
#define TILEY_SHIFT 2

/* Output tile size : 4x4 = Each thread computes 16 float values*/
/* Required global threads = (widthC / 4, heightC / 4) */
/* This kernel runs on 7xx and CPU as they don't have hardware local memory */

__kernel void mmmKernel(__global float4 *matrixA,
                        __global float4 *matrixB,
                        __global float4* matrixC,
            uint widthA)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));


    float4 sum0 = (float4)(0);
    float4 sum1 = (float4)(0);
    float4 sum2 = (float4)(0);
    float4 sum3 = (float4)(0);

    /* Vectorization of input Matrices reduces their width by a factor of 4 */
     uint widthB = widthA /4;

    for(int i = 0; i < widthA; i=i+4)
    {
        float4 tempA0 = matrixA[i/4 + (pos.y << TILEY_SHIFT) * (widthA / 4)];
        float4 tempA1 = matrixA[i/4 + ((pos.y << TILEY_SHIFT) + 1) * (widthA / 4)];
        float4 tempA2 = matrixA[i/4 + ((pos.y << TILEY_SHIFT) + 2) * (widthA / 4)];
        float4 tempA3 = matrixA[i/4 + ((pos.y << TILEY_SHIFT) + 3) * (widthA / 4)];

        //Matrix B is not transposed 
        float4 tempB0 = matrixB[pos.x + i * widthB];	
        float4 tempB1 = matrixB[pos.x + (i + 1) * widthB];
        float4 tempB2 = matrixB[pos.x + (i + 2) * widthB];
        float4 tempB3 = matrixB[pos.x + (i + 3) * widthB];

        sum0.x += tempA0.x * tempB0.x + tempA0.y * tempB1.x + tempA0.z * tempB2.x + tempA0.w * tempB3.x;
        sum0.y += tempA0.x * tempB0.y + tempA0.y * tempB1.y + tempA0.z * tempB2.y + tempA0.w * tempB3.y;
        sum0.z += tempA0.x * tempB0.z + tempA0.y * tempB1.z + tempA0.z * tempB2.z + tempA0.w * tempB3.z;
        sum0.w += tempA0.x * tempB0.w + tempA0.y * tempB1.w + tempA0.z * tempB2.w + tempA0.w * tempB3.w;

        sum1.x += tempA1.x * tempB0.x + tempA1.y * tempB1.x + tempA1.z * tempB2.x + tempA1.w * tempB3.x;
        sum1.y += tempA1.x * tempB0.y + tempA1.y * tempB1.y + tempA1.z * tempB2.y + tempA1.w * tempB3.y;
        sum1.z += tempA1.x * tempB0.z + tempA1.y * tempB1.z + tempA1.z * tempB2.z + tempA1.w * tempB3.z;
        sum1.w += tempA1.x * tempB0.w + tempA1.y * tempB1.w + tempA1.z * tempB2.w + tempA1.w * tempB3.w;

        sum2.x += tempA2.x * tempB0.x + tempA2.y * tempB1.x + tempA2.z * tempB2.x + tempA2.w * tempB3.x;
        sum2.y += tempA2.x * tempB0.y + tempA2.y * tempB1.y + tempA2.z * tempB2.y + tempA2.w * tempB3.y;
        sum2.z += tempA2.x * tempB0.z + tempA2.y * tempB1.z + tempA2.z * tempB2.z + tempA2.w * tempB3.z;
        sum2.w += tempA2.x * tempB0.w + tempA2.y * tempB1.w + tempA2.z * tempB2.w + tempA2.w * tempB3.w;

        sum3.x += tempA3.x * tempB0.x + tempA3.y * tempB1.x + tempA3.z * tempB2.x + tempA3.w * tempB3.x;
        sum3.y += tempA3.x * tempB0.y + tempA3.y * tempB1.y + tempA3.z * tempB2.y + tempA3.w * tempB3.y;
        sum3.z += tempA3.x * tempB0.z + tempA3.y * tempB1.z + tempA3.z * tempB2.z + tempA3.w * tempB3.z;
        sum3.w += tempA3.x * tempB0.w + tempA3.y * tempB1.w + tempA3.z * tempB2.w + tempA3.w * tempB3.w;
    }
    matrixC[pos.x + ((pos.y <<  TILEY_SHIFT) + 0) * widthB] = sum0;
    matrixC[pos.x + ((pos.y <<  TILEY_SHIFT) + 1) * widthB] = sum1;
    matrixC[pos.x + ((pos.y <<  TILEY_SHIFT) + 2) * widthB] = sum2;
    matrixC[pos.x + ((pos.y <<  TILEY_SHIFT) + 3) * widthB] = sum3;
}

