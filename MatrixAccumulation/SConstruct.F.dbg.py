#import OclBuilder 
from OclBuilder import build

# GPU-accelerated or reference implementation?
build('mataccF_dbg', ['../OpenCLIntegration/OclWrapperF.cc','matacc_with_wrapper_dbg.cc'])

  

