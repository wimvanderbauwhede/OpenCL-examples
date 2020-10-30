// OpenCL kernel from Apple HelloWorld example
/*
 * I'm using this as a testbed for OpenCL capabilities
 * 
 * */

// This seems to work after a fashion
#define W 16
#define H 16
//#define BSZ 256 //(W*H)
#define BSZ (W*H)

//Surprisingly, this works too
typedef short PixVal; 
typedef float Entropy;
typedef struct {
    PixVal threshold;
    Entropy entropy;
} ThresholdEntropy ;

// This works OK
float compute(float val);
float getVal(__global float*, int ) ;

short addArrayG(__local short**);

__kernel void square(                                                       
   __global float* input,                                              
   __global float* output,                                             
   const unsigned int count)                                           
{                                                                      
   int i = get_global_id(0); 
    __local short a[W][H];
    short tot=0;
    int j,k;
    for (j=0;j<W;j++) {
        for (k=0;k<H;k++) {
        a[j][k]=j;
        tot+=j;
        }
    }
    short res=addArrayG(a);
    res-=tot;
    output[i] = res+compute( getVal(input,i) );                                
    output[i] = input[i] * input[i];                                
}   

short addArrayG(__local short** a) {
    short tot=0;
    int i;
    int j,k;
      for (j=0;j<W;j++) {
          for (k=0;k<H;k++) {
          short tmp = a[j][k];
          tot+=tmp;
          }
      }    
    return tot;
}

float compute(float val) {

    ThresholdEntropy te;
    te.threshold=2;
    te.entropy=1.0;
    return (te.threshold-te.entropy)*val*val;

}
float getVal(__global float* input, int i) {
	return input[i];
}


