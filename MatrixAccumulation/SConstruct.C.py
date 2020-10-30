# Proof of concept SConstruct for integrating C++ into FORTRAN
from OclBuilder import initOcl
envC=Environment(useF=1)
envC=initOcl(envC)
#cxxsources=['testCxx.cc']
csources=['matacc_with_F_wrapper.c']

#envC=Environment(CXX='g++');
#envC.Library('testCxx',cxxsources)

envC['CC']='gcc'
envC.Program('mataccC',csources,LIBS=['OclWrapperF','OclWrapper','stdc++','OpenCL'],LIBPATH='.')
