my @bufszs=(1,2,4,8,16,32,64,128,256);
for my $bufsz (@bufszs) {
my $bs=$bufsz*1024*1024;
print "$bufsz\n";
   system("./BufferBandwidth -pcie -nb $bs | grep PCIe");
}
