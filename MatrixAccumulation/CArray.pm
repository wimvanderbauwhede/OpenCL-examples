package CArray

=pod
I want a C array accessible in Perl, and so that I can pass it to OpenCL as a pointer. So we want a function

void* createArray(int sz) {
    void* ptr = malloc(sz); return ptr;
}

We want to access this array in C, 

   getFloat(float* ptr, int idx) {
       return ptr[idx];
   }

and similar for other types.

Also:

   void putFloat(float* ptr, int idx, float v) {
       ptr[idx]=v;
   }

(It would be nice to do this with C++ templates)

and of course we want to free the memory:

void destroyArray(void* ptr) {
    free(ptr);
}

In Perl, we want to say:

my $carray = new CArray($type, $sz);

$carray->[$i]=$v;
my $w = $carray->[$i];

So we need Tie::Array and code very similar to Tie::Array::Pointer, the only difference I guess is that we have the type and 
based on the type a given/when selection for the access etc.

This means that the object must be [$ptr,$size,$type], so in the OclWrapper the buffer commands need somehow to detect that this is an object, maybe 

UNIVERSAL::can($r,'isa') is the test to see if it is an object or not. To see if it is a proper CArray: if(UNIVERSAL::isa($ref, 'CArray')) {...}

Finally, it would be nice to be able to say:

my $carray = new CArray($type, $perl_array_ref);

which we can do with code similar to the OclWrapper: pack etc.

=cut
