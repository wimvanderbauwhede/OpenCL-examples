#include "modulo.h";


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
    cl_uint number[1];
    cl_uint res[8];
    cl_uint active_thread[1];
    number[0]=16*mWidth-3;
    active_thread[0]=0;
    const cl_uint nmod=8;
   
#if REF!=0
    cl_uint ref_res=number[0] % nmod;
    double tstartref=wsecond();

    double tstopref=wsecond();
#ifdef VERBOSE
    std::cout << "Execution time for reference: "<<(tstopref-tstartref)<<" ms\n";
#else
    std::cout << (tstopref-tstartref); //<<"\n";
#endif
#endif
#if REF!=2
    //--------------------------------------------------------------------------------
    //---- Here starts the actual OpenCL part
    //--------------------------------------------------------------------------------
    cl_int err;                            // error code returned from api calls
    // First check the Platform
    cl::vector<cl::Platform> platformList;
    cl::Platform::get(&platformList);
    checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
//    std::cerr << "Platform number is: " << platformList.size() << std::endl;
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

    unsigned int nunits=info.max_compute_units( devices.front() );

    // Now load the kernel
    // How about a nice class 
    // KernelLoader kl(string filename);
    // cl::Kernel KernelLoader::build(string kernel_name, cl::Context context, cl::Devices devices);
    std::ifstream file("modulo.cl");
    checkErr(file.is_open() ? CL_SUCCESS:-1, "modulo.cl");

    std::string prog(
            std::istreambuf_iterator<char>(file),
            (std::istreambuf_iterator<char>())
            );

    cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length()+1));

    cl::Program program(context, source);
    err = program.build(devices,"");
    checkErr(file.is_open() ? CL_SUCCESS : -1, "Program::build()");
    cl::Kernel kernel(program, "modulo", &err);
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
    cl::Buffer n_buf(
            context,
            CL_MEM_READ_ONLY | CL_MEM_READ_MODE,
            sizeof(cl_uint) ,
            number,
            &err);
    checkErr(err, "Buffer::Buffer input()");

    cl::Buffer at_buf(
            context,
            CL_MEM_READ_ONLY | CL_MEM_READ_MODE,
            sizeof(cl_uint) ,
            active_thread,
            &err);
    checkErr(err, "Buffer::Buffer input()");

    cl::Buffer res_buf(
            context,
            CL_MEM_WRITE_ONLY,
            sizeof(cl_uint)*8,
            NULL,
            &err);
    checkErr(err, "Buffer::Buffer output()");

    // setArg takes the index of the argument and a value of the same type as the kernel argument 
    err = kernel.setArg(0, n_buf );
    checkErr(err, "Kernel::setArg(0)");

    err = kernel.setArg(1, at_buf);
    checkErr(err, "Kernel::setArg(2)");

    err = kernel.setArg(2, res_buf);
    checkErr(err, "Kernel::setArg(2)");

    err = kernel.setArg(3, nmod);
    checkErr(err, "Kernel::setArg(3)");

    // Create the CommandQueue
    cl::CommandQueue queue(context, devices[0], 0, &err);
    checkErr(err, "CommandQueue::CommandQueue()");


    // This is the actual "run" command. The 3 NDranges are
    // offset, global and local    
    cl::Event event;
    double tstart=wsecond();

    for (unsigned int ct=0;ct<number[0];ct++) {
    err = queue.enqueueNDRangeKernel(
            kernel, 
            cl::NullRange,
            cl::NDRange(nmod),
            cl::NDRange(1),
            NULL, 
            &event);
    checkErr(err, "CommandQueue::enqueueNDRangeKernel()");
    event.wait();
    }
    // Read back the results
       err = queue.enqueueReadBuffer(
            res_buf,
            CL_TRUE,
            0,
            sizeof(cl_uint)*8,
            res);
       checkErr(err, "CommandQueue::enqueueReadBuffer()");

    double tstop=wsecond();
#endif
    //--------------------------------------------------------------------------------
    //----  Here ends the actual OpenCL part
    //--------------------------------------------------------------------------------
#ifdef VERBOSE
#if REF==1
//for (unsigned int i=0;i<8;i++) {
//    std::cout << res[i] <<" ";
//}

std::cout <<res[0] <<"<>"<< ref_res<<"\n";
#endif
#if REF!=2
    std::cout << "OpenCL execution time "<<(tstop-tstart)<<" ms\n";
#endif
#else
#if REF!=2
    std::cout << (tstop-tstart);//<<"\n";
#endif
#endif
    return EXIT_SUCCESS;

}
