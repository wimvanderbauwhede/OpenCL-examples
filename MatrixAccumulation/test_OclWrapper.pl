use strict;
use warnings;

use Inline (
'C' => 'DATA', # Don't know why this must be 'DATA'
        LIBS => '-L'.$ENV{'PWD'}.' -lOclWrapperF -lOclWrapper -lstdc++ -lOpenCL',
        INC => '-I'.$ENV{'OPENCL_DIR'}.'/OpenCLIntegration',
	FORCE_BUILD => 1,
        CLEAN_AFTER_BUILD => 0
);

my $ocl =oclInit('matacc.cl','mataccKernel1');
print $ocl,"\n";
my $nunits = oclGetMaxComputeUnits($ocl);
print $nunits,"\n";

#sub  oclInit {
#	(my $srcstr,my $kstr) = @_;
#	my $ocl = initocl($srcstr,$kstr);
#	return $ocl;
#}

#sub oclGetMaxComputeUnits { (my $ocl) = @_;
#	my $nunits = getmaxcomputeunits($ocl);
#	return $nunits;
#}

sub oclMakeReadBuffer {(my $ocl,my $sz) ; 
	my $buf=0;
	return $buf;
}

sub oclMakeWriteBuffer {(my $ocl,my $sz) ; 
	my $buf=0;
	return $buf;
}

sub oclSetFloatArrayArg { (my $ocl,my $idx, my $oclbuf )=@_;
}

sub oclSetIntConstArg { (my $ocl,my $idx, my $val)=@_;
}

sub oclWriteBuffer { (my $ocl,my $oclbuf, my $sz)=@_;
	my $buf=0;
	return $buf;
}

sub oclReadBuffer { (my $ocl,my $oclbuf, my $sz)=@_;
	my $buf=0;
	return $buf;
}

sub runOcl { (my $ocl,my $g, my $l) = @_;
}
__END__
__C__

#include "OclWrapperC.h"

long oclInit(char* clsrcstr, char* kstr) {
	uint64_t ocl_ivp;
    	oclinit_(&ocl_ivp,clsrcstr,kstr);
	return ocl_ivp;
}
int oclGetMaxComputeUnits(long ocl) {
	int nunits;
	oclgetmaxcomputeunits_(&ocl,&nunits);
	return nunits;
}


