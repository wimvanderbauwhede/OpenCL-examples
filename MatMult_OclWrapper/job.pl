#!/usr/bin/perl
my $dev=$ARGV[0] || 'GPU';
my $plat=$ARGV[1] || 'NVIDIA';
my $oclrun='';
if ($dev eq 'CPU') {
	if ($plat eq 'AMD') {
		$oclrun='oclrun -p AMD';
	}
} elsif ($dev eq 'ACC') {
$plat = 'MIC';
} elsif ($dev ne 'GPU') {
	die "$0 [CPU|GPU|ACC]\n";
} else {
	$plat='NVIDIA';
}
my $nruns=5;
#for my $i (0,1..5,7,10) {
for my $i (1..5,7,8,10,11) {
for my $m (0,1,2) {
    print STDERR "KERNEL=$i, MODE=$m\n";
    print "KERNEL=$i, MODE=$m";
    my $r=($i==0)?2:0;
    print STDERR "scons -s -f SConstruct dev=$dev plat=$plat kernel=$i ref=$r v=0 mode=$m nruns=$nruns\n";
    system("scons --warn=no-python-version -s -f SConstruct dev=$dev plat=$plat kernel=$i ref=$r v=0 mode=$m nruns=$nruns");
    print STDERR "$oclrun ./matmult_int_${dev}_${plat}_$i\n";    
    my $res=`$oclrun ./matmult_int_${dev}_${plat}_$i`;
    print "$res";#\t" unless $run==0;
    
    print "\n";
    if ($i==0) {last;}
}
}
