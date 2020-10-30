#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "OclWrapperC.h"
//#include "OclWrapper.h"
// Use a static data size for simplicity
//
//#define WIDTH 1024
// 1024*1024 gives segmentation fault when using static allocation, because it's on the stack
// 32M is the max because CL_DEVICE_MAX_MEM_ALLOC_SIZE = 128MB for my iMac, and a float is 4 bytes.
typedef unsigned int uint;
double wsecond()
{
        struct timeval sampletime;
        double         time;

        gettimeofday( &sampletime, NULL );
        time = sampletime.tv_sec + (sampletime.tv_usec / 1000000.0);
        return( time*1000.0 ); // return time in ms
}

int main(void)
{

	unsigned int i,j,run;
	int nruns=NRUNS;
    // this array uses stack memory because it's declared inside of a function. 
    // So the size of the stack determines the max array size! 
    //float data[DATA_SIZE];              // original data set given to device
    //float results[DATA_SIZE];           // results returned from device


    const uint mSize = WIDTH*WIDTH;
    const uint mWidth = WIDTH;

    // Create the data sets   
    float* mA=(float*)malloc(sizeof(float)*mSize);
    for(i = 0; i < mSize; i++) {
        mA[i] = (float)(1.0/(float)mSize);
    }
#if REF!=0
    float mCref=0.0;
    for (run=1;run<=nruns;run++) {
    	mCref=0.0;
    double tstartref=wsecond();

    for (i = 0; i<mWidth; i++) {
    	for (j = 0; j<mWidth; j++) {
			mCref+=mA[i*mWidth+j];
		}
	}

    double tstopref=wsecond();
#ifdef VERBOSE
    printf("Execution time for reference: %f ms\n",tstopref-tstartref);
#else
    printf("\t%f",tstopref-tstartref);
#endif
    }
#endif
#if REF!=2
    //--------------------------------------------------------------------------------
    //---- Here starts the actual OpenCL part
    //--------------------------------------------------------------------------------
    // Now load the kernel
    int knum= KERNEL;
    const char bkstr[13]="mataccKernel";
//    printf("Kernel base name:<%s>\n",bkstr);
    const char kstr[15]="mataccKernel";
    sprintf(kstr,"%s%d", bkstr, knum);
//    printf("Kernel name:<%s>\n",kstr);
    int64_t ocl_ivp;
    oclinit_(&ocl_ivp,"matacc.cl",kstr);

    int nunits;
    oclgetmaxcomputeunits_(&ocl_ivp,&nunits);
#ifdef VERBOSE
    printf( "Number of compute units: %d\n",nunits);
#endif // VERBOSE

    // Allocate space for results
    float* mC=(float*)malloc(sizeof(float)*nunits);

    // Create the buffers
    int64_t mA_buf_ivp;
    int szA=sizeof(float) * mSize;
    oclmakereadbuffer_(&ocl_ivp,&mA_buf_ivp,&szA); // this results in a new value for mA_buf_ivp

    int64_t mC_buf_ivp;
    int szC=sizeof(float) * nunits;
	oclmakewritebuffer_(&ocl_ivp,&mC_buf_ivp,&szC); // this results in a new value for mA_buf_ivp

	// setArg takes the index of the argument and a value of the same type as the kernel argument
	int pos=0;
    oclsetfloatarrayarg_(&ocl_ivp,&pos, &mA_buf_ivp );
    pos=1;
    oclsetfloatarrayarg_(&ocl_ivp,&pos, &mC_buf_ivp);
    pos=2;
    oclsetintconstarg_(&ocl_ivp,&pos, &mWidth);

    int global,local;

#if KERNEL<7 || KERNEL==14
            global=nunits;
            local=1;
#elif KERNEL<=8 || KERNEL==15
            global=(nunits*16);
            local=16; // 16 threads
#elif KERNEL==9 || KERNEL==16
            global=(nunits*32);
            local=(32); // 32 threads
#elif KERNEL==10
            global=(nunits*16);
            local=(16); // 16 threads
#elif KERNEL==12
            global=(nunits*16);
            local=(16); // 16 threads
#elif KERNEL==13 || KERNEL==17
            global=(nunits*64);
            local=(64); // 64 threads
#elif KERNEL==18 || KERNEL==20
            global=(nunits*128);
            local=(128); // 128 threads
#elif KERNEL==19
            global=(nunits*256);
            local=(256); // 256 threads
#elif KERNEL==11
            global=(nunits);
            local=1; // single thread
#endif

for (run=1;run<=nruns;run++) {

	oclwritebuffer_(&ocl_ivp,&mA_buf_ivp,&szA,mA);

    // This is the actual "run" command.
	double tstart=wsecond();
    runocl_(&ocl_ivp,&global,&local);

    // Read back the results

    oclreadbuffer_(&ocl_ivp,&mC_buf_ivp,&szC,mC);

    double tstop=wsecond();
#endif // REF!=2
    //--------------------------------------------------------------------------------
    //----  Here ends the actual OpenCL part
    //--------------------------------------------------------------------------------
#ifdef VERBOSE
#if REF==1
    float mCtot=0.0;
    for (i=0;i<nunits;i++) {
    	mCtot+=mC[i];
    }
    unsigned int correct=0;               // number of correct results returned
        if(mCtot == mCref) {
            correct++;
        }
    printf("%f<>%f\n", mCtot,mCref);
#endif
#if REF!=2
    printf("OpenCL execution time %f ms\n",tstop-tstart);
} // nruns
#endif
#else
#if REF!=2
	printf("\t%f",tstop-tstart);
} // nruns
#endif
#endif


    free(mA);
#if REF!=2
    free(mC);
#endif
    return EXIT_SUCCESS;

}
