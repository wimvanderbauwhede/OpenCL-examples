#!/usr/bin/perl
my $nruns=50;
for my $i (14,4,5,6) {
	for my $m (0,1) {
	    print STDERR "KERNEL=$i, MODE=$m\n";
	    print "KERNEL=$i, MODE=$m";
	    my $r=($i==0)?2:0;
	    system("scons -s -f SConstruct.tesla dev=CPU plat=AMD kernel=$i ref=$r v=0 mode=$m  nruns=$nruns");
	    
	    my $res=`oclrun -p AMD ./matacc$i`;
	    print "$res";#\t" unless $run==0;
	    
	    print "\n";
	    if ($i==0) {last;}
	}
}
