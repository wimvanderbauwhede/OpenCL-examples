#include "matacc_cuda.h"
#include "matacc_cuda_kernel.cuh"

inline void checkErr(int err, const char * name) {
	if (err != CUDA_SUCCESS) {
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
// 32M is the max because CL_DEVICE_MAX_MEM_ALLOC_SIZE = 128MB for my iMac, and a float is 4 bytes.

int main(void)
{


	int nruns=NRUNS;
    // this array uses stack memory because it's declared inside of a function. 
    // So the size of the stack determines the max array size! 
    //float data[DATA_SIZE];              // original data set given to device
    //float results[DATA_SIZE];           // results returned from device


    const uint mSize = WIDTH*WIDTH;
    const uint mWidth = WIDTH;

    // Create the data sets   
    float* mA_buf, *mC_buf;
    float* mA=(float*)malloc(sizeof(float)*mSize);
    for(unsigned int i = 0; i < mSize; i++) {
        mA[i] = (float)(1.0/(float)mSize);
    }
#if REF!=0
    float mCref=0.0;
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
    cudaDeviceProp deviceProp;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
                device, deviceProp.major, deviceProp.minor);
    }    
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
    
/*    
    int err;                            // error code returned from api calls
    // First check the Platform
    cl::vector<cl::Platform> platformList;
    cl::Platform::get(&platformList);
    checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
//    std::cerr << "Platform number is: " << platformList.size() << std::endl;
    std::string platformVendor;
    platformList[0].getInfo((cl_platform_info) CL_PLATFORM_VENDOR,
            &platformVendor);
//    std::cerr << "Platform is by: " << platformVendor << "\n";

#ifndef CPU
//    std::cout << "\nUsing GPU\n";
    // Use the platform info as input for the Context    
    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platformList[0])(), 0 };
    cl::Context context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &err); // GPU-only 
#else
//    std::cout << "\nUsing CPU\n";
    // Use the platform info as input for the Context    
    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM,
#ifndef OSX        
        (cl_context_properties)(platformList[1])(), 0 }; // HACK! only for Tesla on AMD
#else    
        (cl_context_properties)(platformList[0])(), 0 }; // HACK! only for iMac
#endif
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
*/
//    cudaSetDevice(0);
    unsigned int nunits=deviceProp.multiProcessorCount;//info.max_compute_units( devices.front() );
    std::cout << "Number of compute units: "<<nunits<< "\n";
    float* mC=(float*)malloc(sizeof(float)*nunits);


    // Create the buffers
    cudaMalloc((void**)mA_buf, sizeof(float) * mSize);
    /*
    cl::Buffer mA_buf(
            context,
            CL_MEM_READ_MODE,
            sizeof(float) * mSize,
            NULL,
            &err
            );
    checkErr(err, "Buffer::Buffer input()");
*/
    cudaMalloc((void**)mC_buf, sizeof(float) * nunits);
    /*
    cl::Buffer mC_buf(
            context,
            CL_MEM_WRITE_ONLY,
            sizeof(float)*nunits,
            NULL,
            &err);
    checkErr(err, "Buffer::Buffer output()");

    // setArg takes the index of the argument and a value of the same type as the kernel argument 
    err = kernel.setArg(0, mA_buf );
    checkErr(err, "Kernel::setArg(0)");

    err = kernel.setArg(1, mC_buf);
    checkErr(err, "Kernel::setArg(2)");

    err = kernel.setArg(2, mWidth);
    checkErr(err, "Kernel::setArg(2)");

    // Create the CommandQueue
    cl::CommandQueue queue(context, devices[0], 0, &err);
    checkErr(err, "CommandQueue::CommandQueue()");
*/
//    cl::Event event;
for (int run=1;run<=nruns;run++) {
    cudaMemcpy(mA_buf, mA, sizeof(float)*mSize, cudaMemcpyHostToDevice);
    /*
	err = queue.enqueueWriteBuffer(
    		mA_buf,
    		CL_TRUE,
    		0,
    		sizeof(float)*mSize,
    		mA,
    		NULL,
    		NULL);
    checkErr(err, "CommandQueue::enqueueWriteBuffer()");
*/
    // This is the actual "run" command. The 3 NDranges are
    // offset, global and local    

    mataccKernel14<<<nunits,1>>>( mA_buf, mC_buf, mWidth) ;
/*
    err = queue.enqueueNDRangeKernel(
            kernel, 
            cl::NullRange,
#if KERNEL<7 || KERNEL==14
            cl::NDRange(nunits),
            cl::NDRange(1),
#elif KERNEL<=8 || KERNEL==15
            cl::NDRange(nunits*16),
            cl::NDRange(16), // 16 threads
#elif KERNEL==9 || KERNEL==16
            cl::NDRange(nunits*32),
            cl::NDRange(32), // 32 threads
#elif KERNEL==10
            cl::NDRange(nunits*16),
            cl::NDRange(16), // 16 threads
#elif KERNEL==12
            cl::NDRange(nunits*16),
            cl::NDRange(16), // 16 threads
#elif KERNEL==13 || KERNEL==17
            cl::NDRange(nunits*64),
            cl::NDRange(64), // 64 threads
#elif KERNEL==18 || KERNEL==20
            cl::NDRange(nunits*128),
            cl::NDRange(128), // 128 threads
#elif KERNEL==19
            cl::NDRange(nunits*256),
            cl::NDRange(256), // 256 threads
#elif KERNEL==11
            cl::NDRange(nunits),
            cl::NDRange(1), // single thread
#endif
            NULL, 
            &event);
    checkErr(err, "CommandQueue::enqueueNDRangeKernel()");
*/
    double tstart=wsecond();


    cudaMemcpy(mC_buf, mC, sizeof(float)*nunits, cudaMemcpyDeviceToHost);
  //  event.wait();  // segfaults here!  
/*
    // Read back the results
       err = queue.enqueueReadBuffer(
            mC_buf,
            CL_TRUE,
            0,
            sizeof(float)*nunits,
            mC);
    checkErr(err, "CommandQueue::enqueueReadBuffer()");
*/
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
    cudaFree(mA_buf); 
#if REF!=2
    free(mC);
    cudaFree(mC_buf);
#endif
    return EXIT_SUCCESS;

}
