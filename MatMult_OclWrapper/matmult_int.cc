#ifndef OCLV2
#define __CL_ENABLE_EXCEPTIONS
#endif
//#include <functional>
#include <OclWrapper.h>

// Use a static data size for simplicity
//
//#define WIDTH 1024
// 1024*1024 gives segmentation fault when using static allocation, because it's on the stack
// 32M is the max because CL_DEVICE_MAX_MEM_ALLOC_SIZE = 128MB for my iMac, and a cl_int is 4 bytes.

int main(void) {

    // this array uses stack memory because it's declared inside of a function. 
    // So the size of the stack determines the max array size! 
    //cl_int data[DATA_SIZE];              // original data set given to device
    //cl_int results[DATA_SIZE];           // results returned from device


    const uint mSize = WIDTH*WIDTH;
    const uint mWidth = WIDTH;

    // Create the data sets   
    cl_int* mA=(cl_int*)malloc(sizeof(cl_int)*mSize);
    cl_int* mB=(cl_int*)malloc(sizeof(cl_int)*mSize);

   

    for(unsigned int i = 0; i < mSize; i++) {
    	/*
    	// Test matmult with diagonal matrix
    	unsigned int r= i / WIDTH;
    	unsigned int c= i % WIDTH;
    	if (r==c) {
    		mA[i]=r;
    		mB[i]=c;
    	} else {
    		mA[i]=0;
			mB[i]=0;
    	}
    	*/
        mA[i] = rand()%64;//32*2*(cl_int)rand() / (cl_int)RAND_MAX; // average will be 0.5, so WIDTH*0.25. I multiply by 4 values around 1
        //mB[i] = i*1.0/1024.0;//32*2*(cl_int)rand() / (cl_int)RAND_MAX;
        mB[i] = rand() % 64;//32*2*(cl_int)rand() / (cl_int)RAND_MAX;
    }
#if REF!=0
    cl_int* mCref=(cl_int*)malloc(sizeof(cl_int)*mSize);
#if REF!=3
    cl_int mArow[WIDTH];

    double tstartref=wsecond();
/*
 * To make this multi-threaded I must pass the pointers for mA, mB, mC and a th_id to the thread.
 * So I need some struct which I then cast to void* etc.
 * */
    for (uint i = 0; i<mWidth; i++) {
    	// This is an attempt to put a row in the cache.
    	// It sometimes works, giving a speed-up of 5x
    	for (uint j = 0; j<mWidth; j++) {
    		mArow[j]=mA[i*mWidth+j];
    	}
        for (uint j = 0; j<mWidth; j++) {
            cl_int elt=0.0;
            for (uint k = 0; k<mWidth; k++) {
            	elt+=mArow[k]*mB[k*mWidth+j];
//                elt+=mA[i*mWidth+k]*mB[k*mWidth+j];
            }
            mCref[i*mWidth+j]=elt;
        }
    }
    double tstopref=wsecond();
#ifdef VERBOSE
    std::cout << "Execution time for reference: "<<(tstopref-tstartref)<<" ms\n";
#else
//for (int run=0;run<=NRUNS;run++) {
    std::cout << "\t"<<(tstopref-tstartref);//<<"\n";
//}
#endif
#endif // REF!=3

#endif // REF!=0
#if REF!=2
    //--------------------------------------------------------------------------------
    //---- Here starts the actual OpenCL part
    //--------------------------------------------------------------------------------
    cl_int* mC=(cl_int*)malloc(sizeof(cl_int)*mSize);
#ifdef VERBOSE
    std::cout << "Creating OclWrapper\n";
#endif
    OclWrapper ocl;
    unsigned int nunits= NGROUPS;
	if (nunits==0) {
		nunits = ocl.getMaxComputeUnits();
	}
#ifdef VERBOSE
    std::cout << "Number of work groups = "<< nunits <<"\n";
    std::cout << "Compiling Kernel\n";
#endif
#if REF==3
    int knum2= KERNEL;
    for (int knum=1;knum<=knum2;knum+=knum2-1) {
#ifdef VERBOSE
    std::cout << "Computing KERNEL=="<<knum<<"\n";
#endif
#else
	int knum= KERNEL;
#endif
    std::ostringstream sstr; sstr << "matmultKernel"<< knum;
    std::string stlstr=sstr.str();
   const char* kstr = stlstr.c_str();
if (knum==1) {
    ocl.loadKernel("matmult_int.cl",kstr);
} else {
#if KERNEL != 10
//std::cout << "Loading kernel "<<kstr<< " from matmult_int.cl\n";
    ocl.loadKernel("matmult_int.cl",kstr);
#else
    ocl.loadKernel("divide_and_conquer_int.cl",kstr);
#endif    
}

#ifdef VERBOSE
    std::cout << "Creating Buffers\n";
#endif
    // Create the buffers
    cl::Buffer mA_buf= ocl.makeReadBuffer(sizeof(cl_int) * mSize);
    cl::Buffer mB_buf= ocl.makeReadBuffer(sizeof(cl_int) * mSize);
    cl::Buffer mC_buf= ocl.makeWriteBuffer(sizeof(cl_int) * mSize);

    for (int run=1;run<=NRUNS;run++) {
      double tstart=wsecond();
	ocl.writeBuffer( mA_buf, sizeof(cl_int)*mSize, mA);
	ocl.writeBuffer( mB_buf, sizeof(cl_int)*mSize, mB);

#ifdef VERBOSE
    std::cout << "Creating CommandQueue\n";
#endif
    // Create the CommandQueue
    if (knum==1) {
    	ocl.enqueueNDRange (  cl::NDRange (mWidth,mWidth) , cl::NullRange ) ;
    } else {
#if KERNEL==1
 ocl.enqueueNDRange (  cl::NDRange (mWidth,mWidth) , cl::NullRange ) ;
#elif KERNEL==2||KERNEL==3||KERNEL==5
 ocl.enqueueNDRange (  cl::NDRange (mWidth/2,mWidth/2) , cl::NullRange ) ;
#elif KERNEL==4
 ocl.enqueueNDRange (  cl::NDRange (mWidth/4,mWidth/4) , cl::NullRange ) ;
#elif KERNEL==7
 ocl.enqueueNDRange (
            cl::NDRange(mWidth,mWidth),
            cl::NullRange
            );
#elif KERNEL==8
 ocl.enqueueNDRange (  cl::NDRange (mWidth*16), cl::NDRange(16) ) ; // more than 16 results in CL_INVALID_COMMAND_QUEUE on iMac
#elif KERNEL==9
 ocl.enqueueNDRange (  cl::NDRange (mWidth*32/4), cl::NDRange(32) ) ;
#elif KERNEL==10
 ocl.enqueueNDRange (  cl::NDRange (mWidth,mWidth), cl::NDRange(16,16) ) ;
#elif KERNEL==11
 ocl.enqueueNDRange (  cl::NDRange (nunits*NTH), cl::NDRange(NTH) ) ;
#else
 ocl.enqueueNDRange (  cl::NDRange (mWidth,mWidth), cl::NullRange ) ;
std::cout << "Only 1..5, 7, 8, 10 and 11 are supported\n";
exit(1);
#endif
    }
#ifdef VERBOSE
    std::cout << "Running ...\n";
#endif
	 ocl.runKernel(mA_buf, mB_buf,mC_buf,mWidth ).wait();

#ifdef VERBOSE
    std::cout << "... Done!\n";
#endif
    // Read back the results
#if REF==3
    if (knum2!=1 && knum==1) {
#ifdef VERBOSE
	    std::cout << "Using KERNEL==1 as reference\n";
#endif
	    ocl.readBuffer(mC_buf,sizeof(cl_int) * mSize,mCref);
	} else {
		ocl.readBuffer(mC_buf,sizeof(cl_int) * mSize,mC);
	}
#else // REF!=3
    ocl.readBuffer(mC_buf,sizeof(cl_int) * mSize,mC);
#endif // REF==3
#ifdef VERBOSE
    std::cout << "... Read-back done!\n";
#endif

    double tstop=wsecond();

#endif // REF!=2
    //--------------------------------------------------------------------------------
    //----  Here ends the actual OpenCL part
    //--------------------------------------------------------------------------------
#ifdef VERBOSE
#if REF==3
    if (knum2!=1 && knum==knum2) {
#endif

#if REF==1 || REF==3
    unsigned int correct=0;               // number of correct results returned
    int nerrors=0;
    int max_nerrors=mSize;
    for (unsigned int i = 0; i < mSize; i++) {
		int diff = mC[i] - mCref[i];
        if(diff==0) { // 2**-20
            correct++;
        } else {
        	nerrors++;
        	if (nerrors>max_nerrors) break;
        }
    }
    free(mCref);
    if (nerrors==0) {
	    std::cout << "Correct!\n";
    }  else {
	    std::cout << "#errors: "<<nerrors<<"\n";
	    std::cout << "Computed '"<<correct<<"/"<<mSize<<"' correct values!\n";
    }
#endif // REF==1
#if REF!=2
    std::cout << "OpenCL execution time "<<(tstop-tstart)<<" ms\n";
    }
#if REF==3
    }// knum
#endif // REF==3
#endif // REF!=2
#if REF==3
    }
#endif

#else // NOT VERBOSE

#if REF==3
    if (knum2!=1 && knum==knum2) {
#endif
#if REF!=2
    std::cout <<"\t"<< (tstop-tstart);//<<"\n";
} // loop over NRUNS
#if REF==3
}}// knum
#endif // REF==3
#endif // REF!=2
#endif // VERBOSE


    free(mA);
    free(mB);
#if REF!=2
    free(mC);
#endif
    return EXIT_SUCCESS;

}
