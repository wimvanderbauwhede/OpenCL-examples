#!/usr/bin/perl
use strict;
use warnings;

use Inline (
'C' => 'DATA',

# => (Config => 
	LIBS => '-L. -lOclWrapperF -lOclWrapper -lstdc++ -lOpenCL',
	INC => '-I'.$ENV{'OPENCL_DIR'}.'/OpenCLIntegration'
#	)
);

greet('HELLO!');
my $sz=10;
my $a= [map {($_%16)*3.14159/1000} (1..$sz)];
my $ap =  pack('f*',@{$a});
square_floats($ap,$sz);
 # Print the results
 print "Got ", join ("\n", unpack('f*', $ap)), "\n";

__END__
__C__

//#include "OclWrapperC.h"

 void greet(SV* sv_name) {
      printf("Hello %s!\n", SvPV(sv_name, PL_na));
    }
/*
unsigned long long initocl(const char* clsrcstr, const char* kstr) {
	int64_t ocl_ivp;
    	oclinit_(&ocl_ivp,"matacc.cl",kstr);
	return ocl_ivp;
}
int getmaxcomputeunits(uint64_t* ocl) {
	int nunits;
	oclgetmaxcomputeunits_(ocl,&nunits);
	return nunits;
}

*/
void square_floats(char* ap, int sz) {
float* data = (float*) ap;
int i;
for (i=0;i<sz;i++) {
data[i]*=data[i];
}

}

