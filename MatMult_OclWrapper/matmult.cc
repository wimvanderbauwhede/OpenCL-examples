#include <Timing.h>
#include <OclWrapper.h>

// Use a static data size for simplicity
//
//#define WIDTH 1024
// 1024*1024 gives segmentation fault when using static allocation, because it's on the stack
// 32M is the max because CL_DEVICE_MAX_MEM_ALLOC_SIZE = 128MB for my iMac, and a cl_float is 4 bytes.

int main(void) {

    // this array uses stack memory because it's declared inside of a function. 
    // So the size of the stack determines the max array size! 
    //cl_float data[DATA_SIZE];              // original data set given to device
    //cl_float results[DATA_SIZE];           // results returned from device


    const uint mSize = WIDTH*WIDTH;
    const uint mWidth = WIDTH;

    // Create the data sets   
    cl_float* mA=(cl_float*)malloc(sizeof(cl_float)*mSize);
    cl_float* mB=(cl_float*)malloc(sizeof(cl_float)*mSize);

   

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
        mA[i] = 32*2*(cl_float)rand() / (cl_float)RAND_MAX; // average will be 0.5, so WIDTH*0.25. I multiply by 4 values around 1
        //mB[i] = i*1.0/1024.0;//32*2*(cl_float)rand() / (cl_float)RAND_MAX;
        mB[i] = 32*2*(cl_float)rand() / (cl_float)RAND_MAX;
    }
#if REF!=0
    cl_float* mCref=(cl_float*)malloc(sizeof(cl_float)*mSize);
#if REF!=3
    cl_float mArow[WIDTH];

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
            cl_float elt=0.0;
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
for (int run=0;run<=NRUNS;run++) {
    std::cout << "\t"<<(tstopref-tstartref);//<<"\n";
}
#endif
#endif // REF!=3

#endif // REF!=0
#if REF!=2
    //--------------------------------------------------------------------------------
    //---- Here starts the actual OpenCL part
    //--------------------------------------------------------------------------------
    cl_int err;
    cl_float* mC=(cl_float*)malloc(sizeof(cl_float)*mSize);
#ifndef DEV_CPU
    bool useGPU=true;
#else
    bool useGPU=false;
#endif
#ifdef VERBOSE
    std::cout << "Creating OclWrapper\n";
#endif
    OclWrapper ocl(useGPU);
#ifdef VERBOSE
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
    ocl.loadKernel("matmult.cl",kstr);
} else {
#if KERNEL != 10
    ocl.loadKernel("matmult.cl",kstr);
#else
    ocl.loadKernel("divide_and_conquer.cl",kstr);
#endif    
}

#ifdef VERBOSE
    std::cout << "Creating Buffers\n";
#endif
    for (int run=1;run<=NRUNS;run++) {
      double tstart=wsecond();
    // Create the buffers
    cl::Buffer mA_buf(
    		*(ocl.context_p),
            CL_MEM_READ_ONLY | CL_MEM_READ_MODE,
            sizeof(cl_float) * mSize,
            mA,
            &err);
    checkErr(err, "Buffer::Buffer input()");

    cl::Buffer mB_buf(
    		*(ocl.context_p),
            CL_MEM_READ_ONLY | CL_MEM_READ_MODE,
            sizeof(cl_float) * mSize,
            mB,
            &err);
    checkErr(err, "Buffer::Buffer input()");

    cl::Buffer mC_buf(
    		*(ocl.context_p),
            CL_MEM_WRITE_ONLY,
            sizeof(cl_float) * mSize,
            NULL,
            &err);
    checkErr(err, "Buffer::Buffer output()");
    std::cout << "MAX_COMPUTE_UNITS:"<<ocl.deviceInfo.max_compute_units(ocl.devices[ocl.deviceIdx]) <<"\n";
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
//kernel_func =
		ocl.enqueueNDRange (  cl::NDRange (mWidth,mWidth), cl::NDRange(16,16) ) ;
#else
ocl.enqueueNDRange (  cl::NDRange (mWidth,mWidth), cl::NullRange ) ;
std::cout << "Only 1..5, 7, 8 and 10 supported\n";
exit(1);
#endif
    }
#ifdef VERBOSE
    std::cout << "Running ...\n";
#endif

    // This is the actual "run" command. The 3 NDranges are
    // offset, global and local    
//    cl::Event event;
//    event=
    		ocl.kernel_functor( mA_buf, mB_buf, mC_buf, mWidth ).wait();
//    event.wait();
#ifdef VERBOSE
    std::cout << "... Done!\n";
#endif
    // Read back the results
#if REF==3
    if (knum2!=1 && knum==1) {
#ifdef VERBOSE
    std::cout << "Using KERNEL==1 as reference\n";
#endif
    	ocl.readBuffer(mC_buf,sizeof(cl_float) * mSize,mCref);
        } else {
        	ocl.readBuffer(mC_buf,sizeof(cl_float) * mSize,mC);
        }
#else
    ocl.readBuffer(mC_buf,sizeof(cl_float) * mSize,mC);
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
    float max_error=0;
    int max_error_i=0;
    float max_error_ref=0.0;
    float max_error_ocl=0.0;
    for (unsigned int i = 0; i < mSize; i++) {
    	float reldiff = (mC[i] > mCref[i])? (mC[i] - mCref[i])/mCref[i] : (mCref[i] - mC[i])/mCref[i];
        if(reldiff<1.0e-6) { // 2**-20
            correct++;
        } else {
        	if (reldiff > max_error) {
        		max_error=reldiff;
                max_error_i=i;
                max_error_ref=mCref[i];
                max_error_ocl=mC[i];
        	}
//        	std::cout << i <<" ("<<i/WIDTH<<","<<i%WIDTH<<"):"<<mC[i] <<"!="<< mCref[i] <<" (delta="<<mC[i]-mCref[i]<<",reldiff="<< reldiff  <<")\n";
        	nerrors++;
        	if (nerrors>max_nerrors) break;
        }
    }
    std::cout << "Max. rel. error: "<<max_error*100<<"%\n";
    std::cout << max_error_i<<":"<<max_error_ref<<"<>"<<max_error_ocl<<"\n";
    free(mCref);
    std::cout << "Computed '"<<correct<<"/"<<mSize<<"' correct values!\n";
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
}
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
