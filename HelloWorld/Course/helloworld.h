/*
 * helloworld.h
 *
 *  Created on: June 16, 2011
 *      Author: wim
 */

#ifndef HELLO_H_
#define HELLO_H_
#include <iostream>
#include <fstream>

#define __NO_STD_VECTOR // Use cl::vector instead of STL version

#ifdef OSX
#include <cl.hpp>
#else
#include <CL/cl.hpp>
#endif


inline void checkErr(cl_int err, const char * name);

#endif /* HELLO_H_ */
