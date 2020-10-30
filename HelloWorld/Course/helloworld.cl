// Keep Eclipse happy
#ifdef __CDT_PARSER__
#define __global
#define __local
#define __private
#define __kernel
#endif
__kernel void hello_world ( __global char* str,__global char* c1, __global char* c2) {
//    __private const char hello_world_str[14]={'H','e','l','l','o',',',' ','W','o','r','l','d','!','\n'};
    // This does not work (empty string)
    //__private const uchar* hello_world_str={'H','e','l','l','o',',',' ','W','o','r','l','d','!','\n'};
    // This gives a compiler error
    //__private const char* hello_world_str= "Hello, World!\n";
    // This gives a compiler error (OK for Intel OpenCL 27 June 2013)
    __private const char hello_world_str[14]= "Hello, World!\n";
    for (__private int i=0;i<14;i++) 
        str[i]=hello_world_str[i];
/*
    __private int i=0;
    while (i<14) { 
        str[i]=hello_world_str[i];
        i++;
    }
*/
    str[14]=*c2;
    str[15]=*c2;
    str[16]=0;
}
__kernel void hello_world_1arg ( __global char* str) {
    __private const char hello_world_str[14]={'H','e','l','l','o',',',' ','W','o','r','l','d','!','\n'};
    // This does not work (empty string)
    //__private const uchar* hello_world_str={'H','e','l','l','o',',',' ','W','o','r','l','d','!','\n'};
    // This gives a compiler error
    //__private const char* hello_world_str= "Hello, World!\n";
    // This gives a compiler error
    //__private const char hello_world_str[14]= "Hello, World!\n";
    //for (int i=0;i<14;i++) 
    //    str[i]=hello_world_str[i];
    __private int i=0;
    while (i<14) { 
        str[i]=hello_world_str[i];
        i++;
    }
    str[14]=0;str[15]=0;str[16]=0;
}
