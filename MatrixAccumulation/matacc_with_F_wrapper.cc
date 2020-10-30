#include "OclWrapperF.h"
//#include "OclWrapper.h"
// Use a static data size for simplicity
//
//#define WIDTH 1024
// 1024*1024 gives segmentation fault when using static allocation, because it's on the stack
// 32M is the max because CL_DEVICE_MAX_MEM_ALLOC_SIZE = 128MB for my iMac, and a cl_float is 4 bytes.

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
    bool use_gpu=true;
#ifdef CPU
    use_gpu=false;
#endif
    // Now load the kernel
    int knum= KERNEL;
    std::ostringstream sstr; sstr << "mataccKernel"<< knum;
    std::string stlstr=sstr.str();
    const char* kstr = stlstr.c_str();

    int64_t ocl_ivp;
    oclinit_(&ocl_ivp,"matacc.cl",kstr);

    int nunits;
    oclgetmaxcomputeunits_(&ocl_ivp,&nunits);
#ifdef VERBOSE
    std::cout << "Number of compute units: "<<nunits<< "\n";
#endif // VERBOSE

    // Allocate space for results
    cl_float* mC=(cl_float*)malloc(sizeof(cl_float)*nunits);

    // Create the buffers
    int64_t mA_buf_ivp;
    oclmakereadbuffer_(&ocl_ivp,&mA_buf_ivp,sizeof(cl_float) * mSize); // this results in a new value for mA_buf_ivp

    int64_t mC_buf_ivp;
	oclmakewritebuffer_(&ocl_ivp,&mC_buf_ivp,sizeof(cl_float) * nunits); // this results in a new value for mA_buf_ivp

	// setArg takes the index of the argument and a value of the same type as the kernel argument
    oclsetfloatarrayarg_(&ocl_ivp,0, &mA_buf_ivp );
    oclsetfloatarrayarg_(&ocl_ivp,1, &mC_buf_ivp);
    oclsetintconstarg_(&ocl_ivp,2, mWidth);


for (int run=1;run<=nruns;run++) {
	oclwritebuffer_(&ocl_ivp,&mA_buf_ivp,sizeof(cl_float) * mSize,mA);

    // This is the actual "run" command.
	double tstart=wsecond();
    runocl_(&ocl_ivp,
#if KERNEL<7 || KERNEL==14
            nunits,
            1
#elif KERNEL<=8 || KERNEL==15
            (nunits*16),
            (16) // 16 threads
#elif KERNEL==9 || KERNEL==16
            (nunits*32),
            (32) // 32 threads
#elif KERNEL==10
            (nunits*16),
            (16) // 16 threads
#elif KERNEL==12
            (nunits*16),
            (16) // 16 threads
#elif KERNEL==13 || KERNEL==17
            (nunits*64),
            (64) // 64 threads
#elif KERNEL==18 || KERNEL==20
            (nunits*128),
            (128) // 128 threads
#elif KERNEL==19
            (nunits*256),
            (256) // 256 threads
#elif KERNEL==11
            (nunits),
            (1) // single thread
#endif
            );

    // Read back the results
    oclreadbuffer_(&ocl_ivp,&mC_buf_ivp,sizeof(cl_float) * nunits,mC);

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
