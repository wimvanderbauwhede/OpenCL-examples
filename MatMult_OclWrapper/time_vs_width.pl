#!/usr/bin/perl
use warnings;
use strict;
my $dev=$ARGV[0] || 'GPU';
my $plat=$ARGV[1] || 'NVIDIA';
my $oclrun='';
my $nth =128;
if ($dev eq 'CPU') {
	if ($plat eq 'AMD') {
		$oclrun='oclrun -p AMD';
	}
$nth=1;
} elsif ($dev ne 'GPU') {
	die "$0 [CPU|GPU]\n";
} else {
	$plat='NVIDIA';
}
my $nruns= 20;

for my $wp (7..13) {
	my $w=1<<$wp;
#		print STDERR "scons -s -f SConstruct kernel=11 v=0 nruns=$nruns nth=$nth w=$w\n";
		system("scons -s -f SConstruct ref=0 kernel=11 v=0 w=$w nth=$nth nruns=$nruns");
#		print STDERR "$oclrun ./matmult_int_${dev}_${plat}_11\n";    
		my $res=`$oclrun ./matmult_int_${dev}_${plat}_11`;
		print "$w\t$res\n";
}
