#include "OclWrapper.h"

int main () {
    const int strSz=17;
    // Instantiate the OpenCL Wrapper
    OclWrapper ocl;
    // Load the Kernel
    ocl.loadKernel("helloworld.cl","hello_world");
    ocl.createQueue();
    // Create Buffers
    cl::Buffer& str_buf=ocl.makeWriteBuffer(strSz);
    cl::Buffer& c1=ocl.makeReadBuffer(1);
    cl::Buffer& c2=ocl.makeReadBuffer(1);
    char cc1='\n';
    char cc2=' ';
    ocl.writeBuffer(c1,1,&cc1);
    ocl.writeBuffer(c2,1,&cc2);
    // Enqueue the Kernel
    cl::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer> ocl_kernel_functor( *(ocl.kernel_p) );
    //ocl.enqueueNDRange();
    cl::EnqueueArgs eargs(cl::NDRange(1),cl::NullRange);
    // Run the Kernel and wait for it to finish
    ocl_kernel_functor(eargs,str_buf,c1,c2).wait();
	// Read back the result
    char* str=(char*)malloc(strSz);
    ocl.readBuffer(str_buf,strSz,str);
    // Display the result
    std::cout <<str;
    // Exit
    return 1;
}
