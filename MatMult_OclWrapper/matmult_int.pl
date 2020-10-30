#!/usr/bin/perl

# This is a simple example program to illustrate the use of
# OclWrapper.pm, my OpenCL wrapper for Perl

# The kernel accumulates a large array by splitting it over the number of cores
# (c) Wim Vanderbauwhede 2011,2012

use warnings;
use strict;
use feature qw(say);
use Time::HiRes qw(gettimeofday tv_interval);

use OclWrapper;
use CTypes qw(int unsigned int);

# Square matrix dimension
my $WIDTH=1024;
my $mSize = $WIDTH*$WIDTH;
my $mWidth = $WIDTH;

my $REF=0;
my $nruns= 1;

# Create the data sets   ;
my $tc0=[gettimeofday];
# nice but useless for large arrays (anything over 16M hangs)
# my $mA = [ map { 1.0/$mSize } (1..$mSize)];

# *much* faster for large arrays 
# but results in an erroneous "Use of uninitialized value $array in pack" warning
my $mA;
$#{$mA}= $mSize-1;
my $v=1.0/$mSize;
for my $i (0 .. $mSize -1 )  {
    $mA->[$i]=$v;
}

my $mB;
$#{$mB}= $mSize-1;
for my $i (0 .. $mSize -1 )  {
    $mB->[$i]=$v;
}

my $mC;
$#{$mC}= $mSize-1;

my $tc0_tc1=tv_interval ($tc0);
say 'Array creation time: ',$tc0_tc1;


# Accumulate all elements in the matrix
if ($REF==1) {
    my $t0=[gettimeofday];

    for my $run (1..$nruns) {
my $mArow;
$#{$mArow}= $mWidth-1;
        for my $i ( 0 .. $mWidth -1) {

        # This is an attempt to put a row in the cache.
        # It sometimes works, giving a speed-up of 5x
        for my $j (  0 .. $mWidth-1) {
                $mArow->[$j]=$mA->[$i*$mWidth+$j];
        }
            for my $j ( 0 .. $mWidth -1) {
            my $elt=0;
        for my $k ( 0 .. $mWidth -1) {
                $elt+=$mArow->[$k]*$mB->[$k*$mWidth+$j];
            }
            $mC->[$i*$mWidth+$j]=$elt;
        }
    }



    } # run
    my $t0_t1=tv_interval ($t0);
    say 'Pure Perl run time: ',$t0_t1;
} 

#--------------------------------------------------------------------------------;
#---- Here starts the actual OpenCL part;
#--------------------------------------------------------------------------------;
my $t2=[gettimeofday];
# Initialise the OpenCL system;
my $knum= 11; # Kernel number (see .cl source)
my $kstr="matmultKernel$knum";
my $srcstr='matmult_int.cl';
my $ocl = new OclWrapper($srcstr,$kstr);

# This returns the number of cores on the device
$ocl->setNThreads(32);
my $nunits=$ocl->getMaxComputeUnits();
print  "Number of compute units: $nunits\n";

# Create the buffers 
my $mA_buf = $ocl->makeReadBuffer(int, $mSize,$mA); # read from by the kernel
my $mB_buf = $ocl->makeReadBuffer(int, $mSize,$mB); # read from by the kernel
my $mC_buf = $ocl->makeWriteBuffer(int, $mSize,$mC); # written to by the kernel
#say @{$mC};die;
# setArg takes the index of the argument and a value of the same type as the kernel argument;
my $mW_const=$ocl->makeConstArg(unsigned int, $mWidth);

my $t2_t3=tv_interval ($t2);
say 'OpenCL setup time: ',$t2_t3;

my $t4=[gettimeofday];
for my $run (1 .. $nruns) {
# Run the kernel
    $ocl->runKernel($mA_buf,$mB_buf, $mC_buf,$mW_const);
#--------------------------------------------------------------------------------;
#----  Here ends the actual OpenCL part;
#--------------------------------------------------------------------------------;
#    my $mCtot=0.0;
#    for my $i (0 .. $nunits-1) {
#        $mCtot=$mCtot+$mC->[$i];
#    }
#    say 'OpenCL result '.(($mCtot==$mCref)? 'matches':'does not match').' Perl result';
#	if ($mCtot!=$mCref) {
#		say "$mCtot <> $mCref";
#	}

} # nruns;
my $t4_t5=tv_interval ($t4);
say 'OpenCL run time: ',$t4_t5;

