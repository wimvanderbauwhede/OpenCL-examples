#include <functional>
#include "OclWrapper.h"
// Use a static data size for simplicity
//
//#define WIDTH 1024
// 1024*1024 gives segmentation fault when using static allocation, because it's on the stack
// 32M is the max because CL_DEVICE_MAX_MEM_ALLOC_SIZE = 128MB for my iMac, and a cl_float is 4 bytes.
//using namespace cl;
int main(void)
{


	int nruns=NRUNS;
    // this array uses stack memory because it's declared inside of a function. 
    // So the size of the stack determines the max array size! 
    //cl_float data[DATA_SIZE];              // original data set given to device
    //cl_float results[DATA_SIZE];           // results returned from device


    const uint mSize = WIDTH*WIDTH;
    const uint mWidth = WIDTH;

    // Create the data sets   
    cl_float* mA=(cl_float*)malloc(sizeof(cl_float)*mSize);
    for(unsigned int i = 0; i < mSize; i++) {
        mA[i] = (cl_float)(1.0/(float)mSize);
    }
#if REF!=0
    cl_float mCref=0.0;
    for (int run=1;run<=nruns;run++) {
    	mCref=0.0;
    double tstartref=wsecond();

    for (uint i = 0; i<mWidth; i++) {
    	for (uint j = 0; j<mWidth; j++) {
			mCref+=mA[i*mWidth+j];
		}
	}

    double tstopref=wsecond();
#ifdef VERBOSE
    std::cout << "Execution time for reference: "<<(tstopref-tstartref)<<" ms\n";
#else
    std::cout <<"\t"<< (tstopref-tstartref); //<<"\n";
#endif
    }
#endif
#if REF!=2
    //--------------------------------------------------------------------------------
    //---- Here starts the actual OpenCL part
    //--------------------------------------------------------------------------------
    OclWrapper ocl;//(use_gpu,DEVIDX);
    unsigned int nunits=ocl.getMaxComputeUnits();
    //std::cout << "Number of compute units: "<<nunits<< "\n";
#if KERNEL==21
    nunits=1;
#endif    
    cl_float* mC=(cl_float*)malloc(sizeof(cl_float)*nunits);
    // Now load the kernel
    int knum= KERNEL;
    std::ostringstream sstr; sstr << "mataccKernel"<< knum;
    std::string stlstr=sstr.str();
    const char* kstr = stlstr.c_str();

    ocl.loadKernel("matacc.cl",kstr);
    ocl.createQueue();
    // Create the buffers
    cl::Buffer& mA_buf= ocl.makeReadBuffer(sizeof(cl_float) * mSize);
    cl::Buffer& mC_buf=ocl.makeWriteBuffer(sizeof(cl_float) * nunits);
//    const int& mWidth_r=mWidth;
for (int run=1;run<=nruns;run++) {
	ocl.writeBuffer(
    		mA_buf,
    		sizeof(cl_float)*mSize,
    		mA
    		);

    // This is the actual "run" command. The 3 NDranges are
    // offset, global and local    

//    cl::make_kernel<cl::Buffer,cl::Buffer,cl_int> kernelFunc( *(ocl.kernel_p) );

    //std::function< cl::Event (const cl::EnqueueArgs&, cl::Buffer, cl::Buffer, cl_int) > kernelFunc = cl::make_kernel<cl::Buffer,cl::Buffer,cl_int>( *(ocl.kernel_p) );
    
    /*
    ocl.enqueueNDRange(
    */
    cl::EnqueueArgs eargs( 
#if KERNEL<7 || KERNEL==14
            cl::NDRange(nunits),
            cl::NDRange(1)
#elif KERNEL<=8 || KERNEL==15
            cl::NDRange(nunits*16),
            cl::NDRange(16) // 16 threads
#elif KERNEL==9 || KERNEL==16
            cl::NDRange(nunits*32),
            cl::NDRange(32) // 32 threads
#elif KERNEL==10
            cl::NDRange(nunits*16),
            cl::NDRange(16) // 16 threads
#elif KERNEL==12
            cl::NDRange(nunits*16),
            cl::NDRange(16) // 16 threads
#elif KERNEL==13 || KERNEL==17
            cl::NDRange(nunits*64),
            cl::NDRange(64) // 64 threads
#elif KERNEL==18 || KERNEL==20
            cl::NDRange(nunits*128),
            cl::NDRange(128) // 128 threads
#elif KERNEL==19
            cl::NDRange(nunits*256),
            cl::NDRange(256) // 256 threads
#elif KERNEL==11
            cl::NDRange(nunits),
            cl::NDRange(1) // single thread
#elif KERNEL==21
	    cl::NDRange(1),
	    cl::NDRange(1) 
#endif
            );

/*
    ocl.kernel_functor(mA_buf, mC_buf,mWidth_r ).wait();  // segfaults here!  
*/
    double tstart=wsecond();
//    kernelFunc( cl::EnqueueArgs(cl::NDRange(nunits),cl::NDRange(1)), mA_buf, mC_buf,mWidth_r);
//    cl::enqueueReadBuffer(mC_buf, CL_TRUE, 0, sizeof(cl_float)*nunits, mC);
    cl::make_kernel<cl::Buffer,cl::Buffer,cl_int> kernelFunc( *(ocl.kernel_p) );
    kernelFunc(eargs,mA_buf, mC_buf,mWidth).wait();
    //cl::Event doneKernel = kernelFunc(eargs,mA_buf, mC_buf,mWidth);
    //doneKernel.wait();
    // Read back the results
    
    ocl.readBuffer(
            mC_buf,
            sizeof(cl_float)*nunits,
            mC);

    double tstop=wsecond();
#endif // REF!=2
    //--------------------------------------------------------------------------------
    //----  Here ends the actual OpenCL part
    //--------------------------------------------------------------------------------
#ifdef VERBOSE
#if REF==1
    float mCtot=0.0;
    for (unsigned int i=0;i<nunits;i++) {
    	mCtot+=mC[i];
    }
    unsigned int correct=0;               // number of correct results returned
        if(mCtot == mCref) {
            correct++;
        }
    std::cout << mCtot <<"<>"<< mCref<<"\n";
#endif
#if REF!=2
    std::cout << "OpenCL execution time "<<(tstop-tstart)<<" ms\n";
} // nruns
#endif
#else
#if REF!=2
    std::cout << "\t"<<(tstop-tstart);//<<"\n";
} // nruns
#endif
#endif


    free(mA);
#if REF!=2
    free(mC);
#endif
    return EXIT_SUCCESS;

}
