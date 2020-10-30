#include "matmult.h"

inline void checkErr(cl_int err, const char * name) {
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit( EXIT_FAILURE);
	}
}

double wsecond()
{
        struct timeval sampletime;
        double         time;

        gettimeofday( &sampletime, NULL );
        time = sampletime.tv_sec + (sampletime.tv_usec / 1000000.0);
        return( time*1000.0 ); // return time in ms
}


// Use a static data size for simplicity
//
//#define WIDTH 1024
// 1024*1024 gives segmentation fault when using static allocation, because it's on the stack
// 32M is the max because CL_DEVICE_MAX_MEM_ALLOC_SIZE = 128MB for my iMac, and a cl_float is 4 bytes.

int main(void)
{

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
        mB[i] = 32*2*(cl_float)rand() / (cl_float)RAND_MAX;
    }
#if REF!=0
    cl_float* mCref=(cl_float*)malloc(sizeof(cl_float)*mSize);
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
    std::cout << (tstopref-tstartref);//<<"\n";
#endif
#endif
#if REF!=2
    //--------------------------------------------------------------------------------
    //---- Here starts the actual OpenCL part
    //--------------------------------------------------------------------------------
    cl_int err;                            // error code returned from api calls
    cl_float* mC=(cl_float*)malloc(sizeof(cl_float)*mSize);

    // First check the Platform
    cl::vector<cl::Platform> platformList;
    cl::Platform::get(&platformList);
    checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
//    std::cerr << "Number of platform is: " << platformList.size() << std::endl;
    std::string platformVendor;
    platformList[0].getInfo((cl_platform_info) CL_PLATFORM_VENDOR,
            &platformVendor);
//    std::cerr << "Platform is by: " << platformVendor << "\n";

    // Use the platform info as input for the Context    
    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platformList[0])(), 0 };
#ifndef CPU
//    std::cout << "\nUsing GPU\n";
    cl::Context context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &err); // CPU-only 
#else
//    std::cout << "\nUsing CPU\n";
    cl::Context context(CL_DEVICE_TYPE_CPU, cprops, NULL, NULL, &err); // CPU-only 
#endif 
    checkErr(err, "Context::Context()");

    cl::vector<cl::Device> devices;
    devices = context.getInfo<CL_CONTEXT_DEVICES>();
    checkErr( devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

    // Get info
    DeviceInfo info;
#ifdef DEVINFO
    info.show(devices.front());
#endif
    // Now load the kernel
    // How about a nice class 
    // KernelLoader kl(string filename);
    // cl::Kernel KernelLoader::build(string kernel_name, cl::Context context, cl::Devices devices);
    std::ifstream file("matmult.cl");
    checkErr(file.is_open() ? CL_SUCCESS:-1, "matmult.cl");

    std::string prog(
            std::istreambuf_iterator<char>(file),
            (std::istreambuf_iterator<char>())
            );

    cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length()+1));

    cl::Program program(context, source);
    err = program.build(devices,"");
    checkErr(file.is_open() ? CL_SUCCESS : -1, "Program::build()");
#if KERNEL==2    
    cl::Kernel kernel(program, "matmultKernel2", &err);
#elif KERNEL==3
    cl::Kernel kernel(program, "matmultKernel3", &err);
#elif KERNEL==4
    cl::Kernel kernel(program, "matmultKernel4", &err);
#elif KERNEL==5
    cl::Kernel kernel(program, "matmultKernel5", &err);
#elif KERNEL==6
    cl::Kernel kernel(program, "matmultKernel6", &err);
#else
    cl::Kernel kernel(program, "matmultKernel1", &err);
#endif    
    checkErr(err, "Kernel::Kernel()");


#if MRMODE==0
#define CL_MEM_READ_MODE CL_MEM_COPY_HOST_PTR
#elif MRMODE==1
#define CL_MEM_READ_MODE (CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR)
#elif MRMODE==2
#define CL_MEM_READ_MODE CL_MEM_USE_HOST_PTR
#else
#define CL_MEM_READ_MODE CL_MEM_COPY_HOST_PTR
#endif

    // Create the buffers
    cl::Buffer mA_buf(
            context,
            CL_MEM_READ_ONLY | CL_MEM_READ_MODE,
            sizeof(cl_float) * mSize,
            mA,
            &err);
    checkErr(err, "Buffer::Buffer input()");

    cl::Buffer mB_buf(
            context,
            CL_MEM_READ_ONLY | CL_MEM_READ_MODE,
            sizeof(cl_float) * mSize,
            mB,
            &err);
    checkErr(err, "Buffer::Buffer input()");

    cl::Buffer mC_buf(
            context,
            CL_MEM_WRITE_ONLY,
            sizeof(cl_float) * mSize,
            NULL,
            &err);
    checkErr(err, "Buffer::Buffer output()");

    // setArg takes the index of the argument and a value of the same type as the kernel argument 
    err = kernel.setArg(0, mA_buf );
    checkErr(err, "Kernel::setArg(0)");

    err = kernel.setArg(1, mB_buf );
    checkErr(err, "Kernel::setArg(1)");

    err = kernel.setArg(2, mC_buf);
    checkErr(err, "Kernel::setArg(2)");

    err = kernel.setArg(3, mWidth); 
    checkErr(err, "Kernel::setArg(2)");

    // Create the CommandQueue
    cl::CommandQueue queue(context, devices[0], 0, &err);
    checkErr(err, "CommandQueue::CommandQueue()");


    // This is the actual "run" command. The 3 NDranges are
    // offset, global and local    
    cl::Event event;
    double tstart=wsecond();
    err = queue.enqueueNDRangeKernel(
            kernel, 
            cl::NullRange,
#if KERNEL==2
            cl::NDRange(mWidth/2,mWidth/2),
#elif KERNEL==3
            cl::NDRange(mWidth/2,mWidth/2),
#elif KERNEL==4
            cl::NDRange(mWidth/4,mWidth/4),
#elif KERNEL==5
            cl::NDRange(mWidth/2,mWidth/2),
#elif KERNEL==6
            cl::NDRange(mWidth/2,mWidth/4),
#else
            cl::NDRange(mWidth,mWidth),
#endif
#if KERNEL==6
            cl::NDRange(1,1),
#else
            cl::NullRange,
#endif
//            cl::NDRange(8,8), 
            NULL, 
            &event);
    checkErr(err, "CommandQueue::enqueueNDRangeKernel()");



    event.wait();  // segfaults here!  

    // Read back the results
       err = queue.enqueueReadBuffer(
            mC_buf,
            CL_TRUE,
            0,
            sizeof(cl_float) * mSize,
            mC);
    checkErr(err, "CommandQueue::enqueueReadBuffer()");

    double tstop=wsecond();
#endif
    //--------------------------------------------------------------------------------
    //----  Here ends the actual OpenCL part
    //--------------------------------------------------------------------------------
#ifdef VERBOSE
#if REF==1
    unsigned int correct;               // number of correct results returned
    int nerrors=0;
    int max_nerrors=mSize;
    float max_error=0;
    for (unsigned int i = 0; i < mSize; i++) {
    	float reldiff = (mC[i] > mCref[i])? (mC[i] - mCref[i])/mCref[i] : (mCref[i] - mC[i])/mCref[i];
        if(reldiff<1.0e-6) { // 2**-20
            correct++;
        } else {
        	if (reldiff > max_error) {
        		max_error=reldiff;
        	}
        //	std::cout << i <<" ("<<i/WIDTH<<","<<i%WIDTH<<"):"<<mC[i] <<"!="<< mCref[i] <<" (delta="<<mC[i]-mCref[i]<<",reldiff="<< reldiff  <<")\n";
        	nerrors++;
        	if (nerrors>max_nerrors) break;
        }
    }
    std::cout << "Max. error: "<<max_error<<"\n";
    free(mCref);
    std::cout << "Computed '"<<correct<<"/"<<mSize<<"' correct values!\n";
#endif
#if REF!=2
    std::cout << "OpenCL execution time "<<(tstop-tstart)<<" ms\n";
#endif
#else
#if REF!=2
    std::cout << (tstop-tstart);//<<"\n";
#endif
#endif
    free(mA);
    free(mB);
#if REF!=2
    free(mC);
#endif
    return EXIT_SUCCESS;

}
