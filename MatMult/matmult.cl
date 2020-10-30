/*
 A simple float matrix multiplication from scratch, on a 1024x1024 matrix
 The first version computes from global memory
 The second tries to use the local memory
 I have 16KB so I can store 4K floats, means I can load 2 rows and 2 cols at once, that is not much
 The third version uses SIMD, but with 2 rows and 2 cols that is rather limited. I can use float2 bit not float2


First, I need to work out a partitioning strategy, but for 1. this is trivial, for 2. the memory determines it

Then I write it in C-ish , then in proper OpenCL
*/

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

// This version of the kernel is an intermediate on, to debug the use of vector floats
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

// Vectorised 4x4
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

// Vectorised 2x4 with use of local memory to cache colums
__kernel void matmultKernel6 (
    __global float4 *mA,
    __global float2 *mB,
    __global float2 *mC,
    const unsigned int mWidth) {

// Every kernel does 4x4 elements 
// We load the 4 columns (1 column float4) into the local memory once
// Then we compute the 4x4 elements using the local values
	
	int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int i=(pos.y<<2);
    int j2=pos.x;
    unsigned int mWidth4 = mWidth/4;
    unsigned int mWidth2 = mWidth/2;
    // NOTE: with cl::NullRange it should not be __local
    // With cl::NDRange(1,1), __local should be OK but in practice it crashes, likely because the GPU is driving the screen as well
    __local float2 mBcols[1024]; // this is 1024*2*4 or 8K, and it doesn't fit without __local :-(
    
    for (unsigned int k=0;k<mWidth;k+=4) {
    	mBcols[k+0]=mB[(k+0)*mWidth2+j2];
    	mBcols[k+1]=mB[(k+1)*mWidth2+j2];
    	mBcols[k+2]=mB[(k+2)*mWidth2+j2];
    	mBcols[k+3]=mB[(k+3)*mWidth2+j2];
    }

	float2 elt0=(float2)(0.0);
	float2 elt1=(float2)(0.0);
	float2 elt2=(float2)(0.0);
	float2 elt3=(float2)(0.0);
	
	for (unsigned int k=0;k<mWidth;k+=4) {    
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
	
	mC[(i+0)*mWidth2+j2]=elt0;	
	mC[(i+1)*mWidth2+j2]=elt1;
	mC[(i+2)*mWidth2+j2]=elt2;
	mC[(i+3)*mWidth2+j2]=elt3; 
			
}

// Vectorised 4x4 with use of local memory to cache colums
__kernel void matmultKernel7 (
    __global float4 *mA,
    __global float4 *mB,
    __global float4 *mC,
    const unsigned int mWidth) {

// Every kernel does 4x4 elements 
// We load the 4 columns (1 column float4) into the local memory once
// Then we compute the 4x4 elements using the local values
	
	int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int i=(pos.y<<2);
    int j4=pos.x;
    int j=(j4<<2);
    //FIXME! DON'T USE!
    __local float4 mBcols[512]; // this is 512*4*4 or 8K, not enough space somehow for 16K :-(
    for (unsigned int k=0;k<mWidth;k++) {
    	mBcols[k]=mB[k*mWidth+j4];
    }
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

			float4 mB0=mBcols[(k+0)];
			float4 mB1=mBcols[(k+1)];
			float4 mB2=mBcols[(k+2)];
			float4 mB3=mBcols[(k+3)];

//			float4 mB0=mB[(k+0)*mWidth4+j4];
//			float4 mB1=mB[(k+1)*mWidth4+j4];
//			float4 mB2=mB[(k+2)*mWidth4+j4];
//			float4 mB3=mB[(k+3)*mWidth4+j4];
 
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

