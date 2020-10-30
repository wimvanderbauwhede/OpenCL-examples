__kernel void commstestKernel (
    __global int *ct
    ) {
    ct[0]=ct[0]+1;
}

