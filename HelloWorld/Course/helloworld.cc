#include "helloworld.h"
#include <fstream>
#include <cstdlib>

int main () {
    const int strSz=16;
	cl_int err;

	// Create the Platform
	cl::vector<cl::Platform> platformList;
	cl::Platform::get(&platformList);
	checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");

	// Create the Context
	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM,
			(cl_context_properties)(platformList[0])(), 0 };
#ifndef CPU	
	cl::Context context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &err); // GPU-only
#else
	cl::Context context(CL_DEVICE_TYPE_CPU, cprops, NULL, NULL, &err); // CPU-only
#endif
    	checkErr(err, "Context::Context()");

    // Find the Devices
    cl::vector<cl::Device> devices;
    devices = context.getInfo<CL_CONTEXT_DEVICES>();
    checkErr( devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

    // Load the Kernel
    std::ifstream file("helloworld.cl");
    checkErr(file.is_open() ? CL_SUCCESS:-1, "helloworld.cl");

    std::string prog(
        std::istreambuf_iterator<char>(file),
        (std::istreambuf_iterator<char>())
        );

    cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length()+1));

    // Build the Kernel
    cl::Program program(context, source);
    err = program.build(devices,"");
    checkErr(file.is_open() ? CL_SUCCESS : -1, "Program::build()");
    cl::Kernel kernel(program, "hello_world_1arg", &err);
    checkErr(err, "Kernel::Kernel()");


    // Create Buffers
    cl::Buffer str_buf(
	        context,
	        CL_MEM_WRITE_ONLY,
	        strSz,
	        NULL,
	        &err);
	checkErr(err, "Buffer::Buffer()");

    // Set the Kernel arguments
    err = kernel.setArg(0, str_buf);
    checkErr(err, "Kernel::setArg()");

    // Create the Command Queue
    cl::CommandQueue queue(context, devices[0], 0, &err);
    checkErr(err, "CommandQueue::CommandQueue()");

    // Enqueue the Kernel
    cl::Event event;
    err = queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(1),
        cl::NullRange,
        NULL,
        &event);
    checkErr(err, "ComamndQueue::enqueueNDRangeKernel()");

    // Wait for the Kernel to finish
    event.wait();
    // Read back the result
    char* str=(char*)malloc(strSz);
    err = queue.enqueueReadBuffer(
        str_buf,
        CL_TRUE,
        0,
        strSz,
        str);
    // Display the result
    std::cout << "<"<<str<<">";
    // Exit
    return EXIT_SUCCESS;
}

inline void checkErr(cl_int err, const char * name) {
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit( EXIT_FAILURE);
	}
}
