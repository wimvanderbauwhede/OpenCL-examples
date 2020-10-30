#include "OclWrapper.h"
// Use a static data size for simplicity
//
//#define WIDTH 1024
// 1024*1024 gives segmentation fault when using static allocation, because it's on the stack
// 32M is the max because CL_DEVICE_MAX_MEM_ALLOC_SIZE = 128MB for my iMac, and a cl_float is 4 bytes.

int main(void)
{

	int nruns=NRUNS;

    OclWrapper ocl(DEVIDX);
    unsigned int nunits=ocl.getMaxComputeUnits();
    std::ostringstream sstr; sstr << "commstestKernel";
    std::string stlstr=sstr.str();
    const char* kstr = stlstr.c_str();

    ocl.loadKernel("commstest.cl",kstr);

    // Create the buffer
    cl::Buffer mA_buf= ocl.makeReadWriteBuffer(sizeof(cl_int));
    cl::Buffer mC_buf= ocl.makeWriteBuffer(sizeof(cl_int));

    int count=0;
    int* mC = new int;
    int* mA= new int;
    mA=&count;
    double tstart=wsecond();
for (int run=1;run<=nruns;run++) {
	ocl.writeBuffer(
    		mA_buf,
    		sizeof(cl_int),
    		mA
    		);

    ocl.enqueueNDRange(
            cl::NDRange(nunits),
            cl::NDRange(1)
            );


    ocl.kernel_functor(mA_buf).wait();  

    // Read back the results
    ocl.readBuffer(
            mA_buf,
            sizeof(cl_int),
            mA);
//    std::cout << (*mC) << "\n";
//    count = *mA;

} // nruns
    double tstop=wsecond();
    std::cout << "Count: "<<count<<"\n";
    std::cout << "OpenCL execution time "<<(tstop-tstart)<<" ms\n";

    return EXIT_SUCCESS;

}
