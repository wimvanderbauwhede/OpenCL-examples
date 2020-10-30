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
int nunits;
int64_t ocl_ivp;
//    ocl=0;
oclinit_(&ocl_ivp,"matacc.cl",kstr);
oclgetmaxcomputeunits_(&ocl_ivp,&nunits);
std::cout << "Number of compute units: "<<nunits<< "\n";
//std::cout <<"INIT OK:"<<ocl_ivp<<"\n";

//F    OclWrapper ocl(use_gpu,DEVIDX);
//F
//F	unsigned int nunits=ocl.getMaxComputeUnits();

    cl_float* mC=(cl_float*)malloc(sizeof(cl_float)*nunits);
    // Now load the kernel
//F    int knum= KERNEL;
//F    std::ostringstream sstr; sstr << "mataccKernel"<< knum;
//F    std::string stlstr=sstr.str();
//F    const char* kstr = stlstr.c_str();
//F
//F    ocl.loadKernel("matacc.cl",kstr);

    // Create the buffers

    int64_t mA_buf_ivp;
/*
 void oclmakereadbuffer_(OclWrapperF ocl_ivp,OclBufferF buf_ivp, int sz) {
	OclWrapper* ocl_p = fromWord(*ocl_ivp);
	int err;
	 cl::Buffer* buf_p= new cl::Buffer(
	            *(ocl_p->context_p),
	            CL_MEM_READ_ONLY,
	            sz,NULL,&err);
	 checkErr(err, "makeReadBuffer()");
	void* buf_vp=static_cast<void*>(&buf_p);
	int64_t* buf_ip=(int64_t*)buf_vp;
	*buf_ivp=(int64_t)buf_ip;
}
 */
    OclWrapper* ocl_p2 = fromWord<OclWrapper*>(ocl_ivp);


//    int64_t* mA_buf_ivpa = &mA_buf_ivp;
//    int64_t* ocl_ivpa = &ocl_ivp;

    oclmakereadbuffer_(&ocl_ivp,&mA_buf_ivp,sizeof(cl_float) * mSize); // this results in a new value for mA_buf_ivp

//    int64_t* mA_buf_ip2=(int64_t*) mA_buf_ivp; // we turn the value into a pointer
//	void* mA_buf_vp=(void*)mA_buf_ip2; // we cast that pointer to void*
//
//    cl::Buffer* mA_buf_p=(cl::Buffer*)mA_buf_vp;

//    cl::Buffer* mA_buf_p=fromWord<cl::Buffer*>(mA_buf_ivp);
//    cl::Buffer& mA_buf=*mA_buf_p;
//    std::cout <<    &mA_buf << "\n";
//    OclWrapper& ocl=*ocl_p2;
//std::cout <<"MAKE WRITE BUFFER\n";
	int64_t mC_buf_ivp;
	oclmakewritebuffer_(&ocl_ivp,&mC_buf_ivp,sizeof(cl_float) * nunits); // this results in a new value for mA_buf_ivp
//    cl::Buffer mC_buf=ocl.makeWriteBuffer(sizeof(cl_float) * nunits);

//    cl::Buffer* mC_buf_p=fromWord<cl::Buffer*>(mC_buf_ivp);
//    cl::Buffer& mC_buf=*mC_buf_p;
//    const int& mWidth_r=mWidth;

    // setArg takes the index of the argument and a value of the same type as the kernel argument
    oclsetfloatarrayarg_(&ocl_ivp,0, &mA_buf_ivp );
    oclsetfloatarrayarg_(&ocl_ivp,1, &mC_buf_ivp);
    oclsetintconstarg_(&ocl_ivp,2, mWidth);


for (int run=1;run<=nruns;run++) {

//	std::cout <<"WRITE BUFFER\n";
	oclwritebuffer_(&ocl_ivp,&mA_buf_ivp,sizeof(cl_float) * mSize,mA); // OK!!!
//	ocl.writeBuffer(
//    		mA_buf,
//    		sizeof(cl_float)*mSize,
//    		mA
//    		);
//	std::cout <<"RUN KERNEL\n";
    // This is the actual "run" command. The 3 NDranges are
    // offset, global and local    
/*
    ocl.enqueueNDRange(
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
#endif
            );

*/
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

//    ocl.kernel_functor(mA_buf, mC_buf,mWidth_r ).wait();

    // Read back the results
    oclreadbuffer_(&ocl_ivp,&mC_buf_ivp,sizeof(cl_float) * nunits,mC);
//    ocl.readBuffer(
//            mC_buf,
//            sizeof(cl_float)*nunits,
//            mC);

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
