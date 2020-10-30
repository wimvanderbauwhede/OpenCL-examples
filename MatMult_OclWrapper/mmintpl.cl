/*
   in each WG g_id, we loop over part of a row, the index is th_range*th_id+ii, for all k
   in each thread, we loop over part of a column, the col index is wg_range*g_id+jj, for all k
   k loops 0 .. mWidth-1
   so we have A[(th_range*th_id+ii)*mWidth+k]
   and B[ wg_range*g_id+jj + k*mWidth]
*/
__kernel void mmKernel11 (
    __global uint *mA,
    __global uint *mB,
    __global uint *mC,
    unsigned int mWidth) {

    uint wg_id=get_group_id(0); 
    uint nunits = get_num_groups(0);
    uint wg_range = mWidth/nunits;
	uint th_id = get_local_id(0);
	uint n_threads=get_local_size(0);
	uint th_range = mWidth/n_threads;
    for (uint jj = wg_id*wg_range;jj<(wg_id+1)*wg_range;jj++) { // loop over part of a row
    // For every thread, loop over part of a col
        for (unsigned int ii=th_id*th_range;ii<(th_id+1)*th_range;ii++) {
            uint elt=0;
            for (unsigned int k=0;k<mWidth;k++) {
                elt+=mA[(th_range*th_id+ii)*mWidth+k]*mB[k*mWidth+wg_range*wg_id+jj];
            }
            mC[ii*mWidth+jj]=elt;
        }
    }
}
