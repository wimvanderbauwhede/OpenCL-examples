#include <functional>
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
        mA[i] = rand()%64;
        mB[i] = rand() % 64;
    }
    cl_int* mCref=(cl_int*)malloc(sizeof(cl_int)*mSize);
    cl_int mArow[WIDTH];
#ifdef REF
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
            }
            mCref[i*mWidth+j]=elt;
        }
    }
    double tstopref=wsecond();
#ifdef VERBOSE
    std::cout << "Execution time for reference: "<<(tstopref-tstartref)<<" ms\n";
#else
    std::cout << "\t"<<(tstopref-tstartref);//<<"\n";
#endif
#endif
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
	int knum= KERNEL;
    std::ostringstream sstr; sstr << "matmultKernel"<< knum;
    std::string stlstr=sstr.str();
   const char* kstr = stlstr.c_str();
   std::cout << "Loading kernel "<<kstr<< " from matmult_int.cl\n";
    ocl.loadKernel("matmult_int.cl",kstr);

#ifdef VERBOSE
    std::cout << "Creating Buffers\n";
#endif
    // Create the buffers
    cl::Buffer mA_buf= ocl.makeReadBuffer(sizeof(cl_int) * mSize);
    cl::Buffer mB_buf= ocl.makeReadBuffer(sizeof(cl_int) * mSize);
    cl::Buffer mC_buf= ocl.makeWriteBuffer(sizeof(cl_int) * mSize);

/*
ocl.kernel.setArg(0,mA_buf);
ocl.kernel.setArg(1,mB_buf);
ocl.kernel.setArg(2,mC_buf);
ocl.kernel.setArg(3,mWidth);
*/

/*
    //cl_int err1;
    cl::make_kernel< cl::Buffer, cl::Buffer, cl::Buffer, cl_uint > runKernel( *(ocl.kernel_p));
    //std::cout << "ERR1: "<<err1<<"\n";
*/

for (int run=1;run<=NRUNS;run++) {
      double tstart=wsecond();
	ocl.writeBuffer( mA_buf, sizeof(cl_int)*mSize, mA);
	ocl.writeBuffer( mB_buf, sizeof(cl_int)*mSize, mB);

#ifdef VERBOSE
    std::cout << "Creating CommandQueue\n";
#endif

#if KERNEL==6 || KERNEL==9 || KERNEL>11
std::cout << "Only 1..5, 7, 8, 10 and 11 are supported\n";
exit(1);
#endif

#ifdef VERBOSE
    std::cout << "Running ...\n";
#endif

 ocl.enqueueNDRange(     
    // Create the CommandQueue
#if KERNEL==1
cl::NDRange(mWidth,mWidth) , cl::NullRange 
#elif KERNEL==2||KERNEL==3||KERNEL==5
 cl::NDRange(mWidth/2,mWidth/2) , cl::NullRange 
#elif KERNEL==4
cl::NDRange(mWidth/4,mWidth/4) , cl::NullRange 
#elif KERNEL==7
cl::NDRange(mWidth,mWidth), cl::NullRange
#elif KERNEL==8
     // more than 16 results in CL_INVALID_COMMAND_QUEUE on iMac
 cl::NDRange(mWidth*16), cl::NDRange(16)  
#elif KERNEL==9
 cl::NDRange(mWidth*32/4), cl::NDRange(32) 
#elif KERNEL==10
cl::NDRange(mWidth,mWidth), cl::NDRange(16,16) 
#elif KERNEL==11
cl::NDRange(nunits*NTH), cl::NDRange(NTH) 
#else
cl::NDRange(mWidth,mWidth), cl::NullRange 
#endif
);

ocl.runKernel(mA_buf, mB_buf,mC_buf,mWidth ).wait();

//	cl::Event evt;

//    cl::make_kernel< cl::Buffer, cl::Buffer, cl::Buffer, cl_int > runKernel( *(ocl.kernel_p) );

//    typedef cl::make_kernel <cl::Buffer&, cl::Buffer&, cl::Buffer&, const cl_int&> KernelType;
//    std::function<KernelType::type_> runKernel = KernelType( *(ocl.kernel_p) );   
//    evt = runKernel(enqArgs, mA_buf, mB_buf, mC_buf, mWidth_r);
/*
cl_int err = ocl.queue_p->enqueueNDRangeKernel( 
        *(ocl.kernel_p), 
        cl::NullRange, 
        cl::NDRange(nunits*NTH), 
        cl::NDRange(NTH) , 
        NULL, 
        &evt);
    checkErr(err, "CommandQueue::enqueueNDRangeKernel()");
    */
//    evt.wait(); 
//    ocl.enqueueNDRangeRun( cl::NDRange(nunits*NTH), cl::NDRange(NTH) );

#ifdef VERBOSE
    std::cout << "... Done!\n";
#endif
    // Read back the results
    ocl.readBuffer(mC_buf,sizeof(cl_int) * mSize,mC);
#ifdef VERBOSE
    std::cout << "... Read-back done!\n";
#endif

    double tstop=wsecond();

    //--------------------------------------------------------------------------------
    //----  Here ends the actual OpenCL part
    //--------------------------------------------------------------------------------
#ifdef VERBOSE

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
    std::cout << "OpenCL execution time "<<(tstop-tstart)<<" ms\n";
} // nruns

#else // NOT VERBOSE
    std::cout <<"\t"<< (tstop-tstart);//<<"\n";
#endif // VERBOSE

    free(mA);
    free(mB);
    free(mC);
    return EXIT_SUCCESS;

}
