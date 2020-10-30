#include "OclWrapper.h"
// Use a static data size for simplicity
//
//#define WIDTH 1024
// 1024*1024 gives segmentation fault when using static allocation, because it's on the stack
// 32M is the max because CL_DEVICE_MAX_MEM_ALLOC_SIZE = 128MB for my iMac, and a cl_float is 4 bytes.

#define SINGLE_TH 0
#define LOG_1 1
#define LOG_2 2
#define PHYS_1 3
#define PHYS_2 4
//#if SELECT != 0
//#define CALC 
//#endif
inline float calc(float x) {
 float v= x;
#ifdef CALC
 for (unsigned int n= 1; n< (1<<(2*SELECT));n++) {
         float fn = (float)n;
     v=v+1.0001/fn;
     v=v/0.9999-1/fn;
 }
#endif
 return v;
}
#define VERBOSE 1
int main(void)
{
    double kernel_exec_time=0.0;
    double kernel_exec_time_nruns=0.0;
    int nruns=NRUNS;

    const uint im = WX;
    const uint jm = WY;
    const uint km = WZ;
    const uint ijkm = im*jm*km;

    // Create the data sets   
    cl_float* mA=(cl_float*)malloc(sizeof(cl_float)*ijkm);
    for(unsigned int i = 0; i < ijkm; i++) {
        mA[i] = 1.0F;//(cl_float)(1.0/(float)ijkm);
    }
#if REF!=0
    cl_float mCref=0.0;
    double tstartref=wsecond();
    for (int run=1;run<=nruns;run++) {
    	mCref=0.0;

    for (uint i = 0; i<im; i++) {
	    for (uint j = 0; j<jm; j++) {
		    for (uint k = 0; k<km; k++) {
                mCref+=calc( mA[i*km*jm+j*km+k] ); 
		    }
	    }
    }
/*
for (uint i=0;i<ijkm;i++) {
mCref+=mA[i];
}
*/
}
    double tstopref=wsecond();
#ifdef VERBOSE
    //std::cout << "Execution time: Reference: "<<(tstopref-tstartref)<<" ms\t";
    std::cout << "\t"<<(tstopref-tstartref)<<"\t";
#else
    std::cout <<"Ref:\t"<< (tstopref-tstartref); //<<"\n";
#endif
    
#endif
#if REF!=2
    //--------------------------------------------------------------------------------
    //---- Here starts the actual OpenCL part
    //--------------------------------------------------------------------------------
    OclWrapper ocl(DEVIDX);
    unsigned int nunits=ocl.getMaxComputeUnits();
		unsigned long int glob_mem_sz = ocl.getGlobalMemSize();
		unsigned long int loc_mem_sz = ocl.getLocalMemSize();
        unsigned long int pref_wg_sz = ocl.getPreferredWorkGroupSizeMultiple();
        unsigned long int pref_nth = ocl.getNThreadsHint();    
    unsigned int nthreads = NTH;
/*    
    std::cout << "nunits: "<< nunits << "\n";    
    std::cout << "global mem sz: "<< glob_mem_sz<< "\n";    
    std::cout << "local mem sz: "<< loc_mem_sz << "\n";    
    std::cout << "pref WG sz: "<< pref_wg_sz << "\n";    
    std::cout << "pref NTH: "<< pref_nth << "\n";    
    */
   // Now load the kernel
#if KERNEL == SINGLE_TH
    std::ostringstream sstr; sstr << "mataccKernel_single_th";
    nunits=1;
    nthreads = 1;

#elif KERNEL == LOG_1
    std::ostringstream sstr; sstr << "mataccKernel_imjmkm";
    nunits=jm*km;
    nthreads = im;
#elif KERNEL == LOG_2
    std::ostringstream sstr; sstr << "mataccKernel_jmkm_im";
    nunits=km;
    nthreads = im;
#elif KERNEL == PHYS_1
    std::ostringstream sstr; sstr << "mataccKernel_lin";
#elif KERNEL == PHYS_2
    std::ostringstream sstr; sstr << "mataccKernel_nunits_nth";
#else 
    std::cerr << "The only valid values for KERNEL are  0 .. 3\n";
    exit(0);
#endif     
    std::string stlstr=sstr.str();
    const char* kstr = stlstr.c_str();

    ocl.loadKernel("matacc.cl",kstr,KERNEL_OPTS);

    //std::cout << "Number of compute units: "<<nunits<< "\n";
    float* mC=(float*)malloc(sizeof(float)*nunits);
    for (unsigned int i = 0;i< nunits;i++ ){
        mC[i]=0.0F;
    }
//    mC[0]=42;
    // Create the buffers
    cl::Buffer mA_buf= ocl.makeReadBuffer(sizeof(float) * ijkm);
    cl::Buffer mC_buf=ocl.makeReadWriteBuffer(sizeof(float) * nunits);
    const int& im_r=im;
    const int& jm_r=jm;
    const int& km_r=km;
        ocl.writeBuffer(
                mA_buf,
                sizeof(cl_float)*ijkm,
                mA
                );

      ocl.writeBuffer(
                mC_buf,
                sizeof(cl_float)*nunits,
                mC
                );
    float mCtot=0.0;

    double tstart=wsecond();
    for (int run=1;run<=nruns;run++) {
    
    ocl.enqueueNDRange(
            cl::NDRange(nunits*nthreads),
            cl::NDRange(nthreads) 
            );

    //cl::Event event = ocl.kernel_functor(mA_buf, mC_buf,im_r ,jm_r,km_r);
    cl::Event event = ocl.runKernel(mA_buf, mC_buf,im_r ,jm_r,km_r);
    event.wait();    
    kernel_exec_time=ocl.getExecutionTime(event);
    kernel_exec_time_nruns+=kernel_exec_time;
    // Read back the results
//    std::cout << "Reading back "<<nunits <<" floats\n";
    ocl.readBuffer(
            mC_buf,
            sizeof(float)*nunits,
            mC);
    mCtot=0.0;
    for (unsigned int i=0;i<nunits;i++) {
    	mCtot+=mC[i];
    }
}
    double tstop=wsecond();
#endif // REF!=2
    //--------------------------------------------------------------------------------
    //----  Here ends the actual OpenCL part
    //--------------------------------------------------------------------------------
#ifdef VERBOSE
#if REF==1
    unsigned int correct=0;               // number of correct results returned
        if(mCtot == mCref) {
            correct++;
        }
    std::cout << "\t"<< mCtot <<"<>"<< mCref<<"\t";
#endif
#if REF!=2
//    std::cout << "OpenCL execution time "<<(tstop-tstart)<<" ms\n";
//
    //std::cout << "\tOpenCL:"<<(tstop-tstart)<<" ms\n";
    std::cout << "\t"<<(tstop-tstart)<<"\t"<< kernel_exec_time_nruns<< "\n";
// } // nruns
#endif
#else
#if REF!=2
    std::cout << "\t"<<(tstop-tstart)<<"\t" << kernel_exec_time_nruns << "\n";
} // nruns
#endif
#endif


    free(mA);
#if REF!=2
    free(mC);
#endif
    return EXIT_SUCCESS;

}
