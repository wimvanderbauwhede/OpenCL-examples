#ifndef __MATMULT_H__
#define __MATMULT_H__
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <sys/time.h>

//For our program, we use a small number of additional C++ headers, which are agnostic to OpenCL.

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

//inline void checkErr(int err, const char * name);
inline double wsecond();
typedef unsigned int uint;

#endif  // __MATMULT_H__

