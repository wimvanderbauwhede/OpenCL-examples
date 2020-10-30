#!/usr/bin/perl 
use warnings;
use strict;
#no strict 'subs';
use CTypes qw(unsigned char int long uint32_t);

my $v1 = unsigned int;
my $v2 = uint32_t;
my $v3 = long ;
my $v4 = long int;

print $v1,"\n";
print $v2,"\n";
print $v3,"\n";
print $v4,"\n";
