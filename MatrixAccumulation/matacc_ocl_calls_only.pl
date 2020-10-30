use OclWrapper;
use CTypes qw(float unsigned int);

# Initialise the OpenCL system;
my $ocl = new OclWrapper('matacc.cl','mataccKernel10');

# This returns the number of cores on the device
my $nunits = $ocl->getMaxComputeUnits();

# Create the buffers 
my $mA_buf = $ocl->makeReadBuffer(float, $mSize); # read from by the kernel
my $mC_buf = $ocl->makeWriteBuffer(float, $nunits); # written to by the kernel

# setArg takes the index of the argument and a value of the same type as the kernel argument;
$ocl->setArrayArg(0, $mA_buf );
$ocl->setArrayArg(1, $mC_buf);
$ocl->setConstArg(2, unsigned int, $mWidth);

# Write the array to the device
$ocl->writeArray($mA_buf,float, $mSize,$mA);
# Run the kernel
$ocl->run($nunits*16,16);
# Read back the results;
my $mC = $ocl->readArray($mC_buf,float, $nunits);

my $mCtot=0.0;
for my $i (0 .. $nunits-1) {
    $mCtot=$mCtot+$mC->[$i];
}

