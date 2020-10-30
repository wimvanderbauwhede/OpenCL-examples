#define CONV_IDX 1
/* clcc
#define NTH 16
#define SELECT 0
#define LOOP_ORDER 1
*/

/*
 A simple float matrix multiplication from scratch, on a 3D matrix

Given:

ijkm = im*jm*km

NDRange = nunits*nthreads

unsigned int idx = get_global_id(0);
// calculate the local range from idx and the full range :
chunk_sz = ijkm/(nunits*nthreads)
// deal with remaineder for last compute unit
// starting index
chunk_idx = chunk_sz*idx
n_tchunks = chunk_sz/nthreads
n_tchunks += (chunk_sz % nthreads == 0) ? 0 : 1

// now we can iterate over this range in chunks of nthreads
t_idx = get_local_id(0)
for tc_idx = 0 .. n_tchunks - 1
// the actual index in the overall array must be
a_idx = chunk_idx + tc_idx*nthreads + t_idx
// From this we can now calc i,j,k
// Compute
end

Alternatively, more simply I suppose, we can create nested loops

unsigned int idx = get_global_id(0);
// From this calc ranges for i,j,k
// Typically im > nunits so we can split i over the compute units

     int gr_id = get_group_id(0); 
     int i_sz = im/nunits;
     int i_start = gr_id*i_sz;
     int i_stop = (gr_id==nunits-1))?im:(gr_id+1)* i_sz;

     int l_id = get_local_id(0);
     int n_tchunks = km/nthreads;
     n_tchunks += (km % nthreads == 0) ? 0 : 1

// We can simply keep j
// k needs to be parallelised over the threads, as it is contiguous
for i = i_start .. i_stop
for j = 1 .. jm
for tc_idx = 0 .. n_tchunks-1
k = tc_idx*nthreads + l_id
// Compute
end
end
end

*/

inline unsigned int FTNREF3D(
                int ix, int jx, int kx,
                        unsigned int iz,unsigned int jz,
                                int i_lb, int j_lb, int k_lb
                                        ) {
        return (iz*jz*(kx-k_lb)+iz*(jx-j_lb)+ix-i_lb);
}

inline unsigned int FTNREF3D0(
                int ix, int jx, int kx,
                        unsigned int iz,unsigned int jz
                                ) {
        return iz*jz*kx+iz*jx+ix ;
}

inline unsigned int FTNREF1D(int ix,int i_lb) {
            return ix-i_lb;
}


// These functions take the lower and upper bounds, rather than the range and the lower bound
inline unsigned int FTNREF3Du(int ix,int jx,int kx,unsigned int i_ub,unsigned int j_ub,int i_lb,int j_lb,int k_lb) {
    return (i_ub - i_lb + 1)*(j_ub - j_lb + 1)*(kx - k_lb)+(i_ub - i_lb + 1)*(jx - j_lb)+(ix - i_lb);
}
// For lower bounds all 0
inline unsigned int FTNREF3Du0(int ix,int jx,int kx,unsigned int i_ub,unsigned int j_ub) {
    return (i_ub + 1)*(j_ub + 1)*kx+(i_ub + 1)*jx+ix;
}



// This is a reusable function to evaluate the effect of the loop ordering

inline int4 calc_loop_iters(int idx, int im, int jm, int km, int i_off, int j_off, int k_off) {
    int4 ijk;
    
#if LOOP_ORDER == 1     
    // jki 
    int ik_range = km*im;
    ijk.s1=  j_off + idx/ik_range; //j
    ijk.s2 = k_off + idx % ik_range / im; //k
    ijk.s0 = i_off + idx % im; //i
#elif LOOP_ORDER == 2
    // ijk: j -> i, k -> j, i-> k
    int kj_range = km*jm;
    ijk.s0=  i_off + idx/kj_range; //i
    ijk.s1 = j_off + idx % kj_range / km; //j
    ijk.s2 = k_off + idx % km; //k
#elif LOOP_ORDER == 3 // This cause an out-or-resources error for Phys1!
    // kij: j -> k, k -> i, i -> j 
    int ji_range = jm*im;
    ijk.s2=  k_off + idx/ji_range; //k
    ijk.s0 = i_off + idx % ji_range / jm; //i
    ijk.s1 = j_off + idx % jm; //j
#elif LOOP_ORDER == 4
    // jik: j -> j, k -> i, i-> k
    int ik_range = km*im;
    ijk.s1=  j_off + idx/ik_range; //j
    ijk.s0 = i_off + idx % ik_range / km; //i
    ijk.s2 = k_off + idx % km; //k
#elif LOOP_ORDER == 5
    // ikj: j->i, k->k,i->j 
    int jk_range = km*jm;
    ijk.s0=  i_off + idx/jk_range; //i
    ijk.s2 = k_off + idx % jk_range / jm; //k
    ijk.s1 = j_off + idx % jm; //j
#elif LOOP_ORDER == 6
    // kji: j->k,k->j,i->i    
    int ij_range = jm*im;
    ijk.s2=  k_off + idx/ij_range; //k
    ijk.s1 = j_off + idx % ij_range / im; //j
    ijk.s0 = i_off + idx % im; //i
#endif    
    return ijk;
}
//================================================================================ 

#if SELECT != 0
#define CALC 
#endif

inline float calc(float x) {
 float v= x;
#ifdef CALC
 const unsigned int n_max = 1<<(2*SELECT);
 for (unsigned int n = 1; n < n_max;n++) {
	 float fn = (float)n;
     v=v+1.0001/fn;
     v=v/0.9999-1/fn;
 }
#endif 
 return v;
}
//================================================================================ 
                          
// Reference
// Single threaded, for debugging
// NDRange global = 1, local = 1;
__kernel void mataccKernel_single_th (
    __global float *mA, 
    __global float *mC,
    const unsigned int im,
    const unsigned int jm,
    const unsigned int km
    ) {
	unsigned int gr_id = get_group_id(0);
    float elt=0.0F;
    unsigned int start = 0;
    unsigned int stop = im*jm*km;
    for (unsigned int i=start;i<stop;i++) {
        	elt+=calc(mA[i]);
    }
    mC[gr_id]=elt;
}


/*
   Logical,1
NDRange = ijkm
local range  = im
unsigned int idx = get_global_id(0);
// calc i,j,k 
// Compute
*/

__kernel void mataccKernel_imjmkm (
    __global float *mA,
    __global float *mC,
    const unsigned int im,
    const unsigned int jm,
    const unsigned int km
    ) {
    __local float elts_th[NTH];
        float elt_g=0.0F;
    unsigned int idx = get_global_id(0);
    unsigned int th_id = get_local_id(0);
	unsigned int gr_id = get_group_id(0);
    int4 ijk = calc_loop_iters(idx,im, jm, km,0,0,0) ;
    unsigned int i=ijk.s0;
    unsigned int j=ijk.s1;
    unsigned int k=ijk.s2;
#ifdef CONV_IDX
    unsigned int f_idx =FTNREF3D0( i, j, k, im,jm);
#else
    unsigned int f_idx = idx
#endif        
    elts_th[th_id]=calc(mA[f_idx]);
    barrier(CLK_LOCAL_MEM_FENCE); // wait for all threads to finish
    
        for (unsigned int ii=0;ii<NTH;ii++) {
            elt_g+=elts_th[ii];
        }
       mC[gr_id]=elt_g;
    
/*
   // Incorrect on GPU!
    float elt_th=calc(mA[idx]);
    
    for (unsigned int ii=0;ii<NTH;ii++) {
        if (ii==th_id) {
            mC[gr_id]=elt_th;
        }
    }
*/

}


// Logical,2
// Linear
// NDRange global = im*km, local = im, loop over jm

__kernel void mataccKernel_jmkm_im (
    __global float *mA, 
    __global float *mC,
    const unsigned int im,
    const unsigned int jm,
    const unsigned int km
    ) {
	unsigned int gl_id = get_global_id(0);
    unsigned int k = get_group_id(0); 
    unsigned int i = get_local_id(0);
   
    __local float elts_th[NTH]; 
    float elts_g=0.0F;
    float elt=0.0F;
    for (unsigned int j=0; j<jm;j++) {
        unsigned int idx = gl_id;
#ifdef CONV_IDX
#if LOOP_ORDER == 1
    unsigned int f_idx =FTNREF3D0( i, j, k, im,jm);
#elif LOOP_ORDER == 2
    unsigned int f_idx =FTNREF3D0( j, i, k, jm,im);
#elif LOOP_ORDER == 3
    unsigned int f_idx =FTNREF3D0( i, k, j, im,km);
#elif LOOP_ORDER == 4
    unsigned int f_idx =FTNREF3D0( j, k, i, jm,km);
#elif LOOP_ORDER == 5
    unsigned int f_idx =FTNREF3D0( k, i, j, km,im);
#elif LOOP_ORDER == 6
    unsigned int f_idx =FTNREF3D0( k, j, i, km,jm);
#endif
#else
    unsigned int f_idx = idx
#endif           
        elt+=calc(mA[f_idx]);
    }
    
	elts_th[i]=elt;
    barrier(CLK_LOCAL_MEM_FENCE);
		for (unsigned int ii=0;ii<NTH;ii++) {
			elts_g+=elts_th[ii];
		}
		mC[k]=elts_g;
	

/* Incorrect on GPU!
    // This is a neat way to sequentialise writes to global memory from each thread
    for (unsigned int ti=0;ti<NTH;ti++) {
        if (ti==i) {
            mC[k]+=elt; // will this work?
        }
    }
*/
}



// Physical,1
// Linear
// NDRange global = nunits*NTH, local = NTH

__kernel void mataccKernel_lin (
    __global float *mA, // although this is constant, setting it to __constant results in CL_OUT_OF_RESOURCES
    __global float *mC,
    const unsigned int im,
    const unsigned int jm,
    const unsigned int km
    ) {
    __local float elts_th[NTH]; 
	unsigned int gr_id = get_group_id(0);
	unsigned int th_id = get_local_id(0);
	unsigned int nunits=get_num_groups(0); // assumes we do this per compute unit,
    float elts_g=0.0F;
    float elt=0.0F;
    unsigned int ijkm=im*jm*km;
    unsigned int gr_range= ijkm/nunits;
    if (gr_id==nunits-1) {
        gr_range += ijkm % nunits;
    }
    unsigned int th_range = gr_range/NTH;
    if (th_id==NTH-1) {
        th_range += gr_range % NTH;
    }
// In every thread
    for (unsigned int th_idx=0;th_idx<th_range;th_idx++) {
        unsigned int idx = gr_id*ijkm/nunits+th_id*th_range+th_idx;
    int4 ijk = calc_loop_iters(idx,im, jm, km,0,0,0) ;
    unsigned int i=ijk.s0;
    unsigned int j=ijk.s1;
    unsigned int k=ijk.s2;
#ifdef CONV_IDX
    unsigned int f_idx =FTNREF3D0( i, j, k, im,jm);
#else
    unsigned int f_idx = idx
#endif           
        	elt+=calc(mA[f_idx]);
    }
    /* Incorrect on GPU
    // This is a neat way to sequentialise writes to global memory from each thread
    for (unsigned int i=0;i<NTH;i++) {
        if (th_id==i) {
            mC[gr_id]+=elt; // will this work?
        }
    }
*/

    	elts_th[th_id]=elt;
    barrier(CLK_LOCAL_MEM_FENCE);
		for (unsigned int ii=0;ii<NTH;ii++) {
			elts_g+=elts_th[ii];
		}
		mC[gr_id]=elts_g;

}

/*
   Physical,2

 NDRange = nunits*NTH

unsigned int idx = get_global_id(0);
// calculate the local range from idx and the full range :
chunk_sz = ijkm/(nunits*nthreads)
// deal with remaineder for last compute unit
// starting index
chunk_idx = chunk_sz*idx
n_tchunks = chunk_sz/nthreads
n_tchunks += (chunk_sz % nthreads == 0) ? 0 : 1

// now we can iterate over this range in chunks of nthreads
t_idx = get_local_id(0)
for tc_idx = 0 .. n_tchunks - 1
// the actual index in the overall array must be
a_idx = chunk_idx + tc_idx*nthreads + t_idx
// From this we can now calc i,j,k
// Compute
end


 */

__kernel void mataccKernel_nunits_nth (
        __global float *mA,
        __global float *mC,
        const unsigned int im,
        const unsigned int jm,
        const unsigned int km
        ) {
    __local float elts_th[NTH]; 
	unsigned int gl_id = get_global_id(0);
	unsigned int gr_id = get_group_id(0);
	unsigned int nunits=get_num_groups(0);
        unsigned int th_id = get_local_id(0);
        float elts_g=0.0F;
        // calculate the local range from idx and the full range :
        unsigned int ijkm = im*jm*km;
        unsigned int chunk_sz = ijkm/nunits;//*NTH);
        unsigned int chunk_idx = chunk_sz*gr_id;
        // Now, this will be too small in some cases
        if (gr_id == nunits-1) {
            if (ijkm % nunits !=0) {
                chunk_sz+=ijkm % nunits;
            }
        }
        // starting index
        unsigned int n_tchunks = chunk_sz/NTH;
        n_tchunks += (chunk_sz % NTH == 0) ? 0 : 1;

        // now we can iterate over this range in chunks of nthreads
        int elt=0;
        for (unsigned int tc_idx = 0 ; tc_idx< n_tchunks; tc_idx++) {
            unsigned int a_idx = chunk_idx + tc_idx*NTH + th_id;
            // From this we can now calc i,j,k
            int4 ijk = calc_loop_iters(a_idx,im, jm, km,0,0,0) ;
            unsigned int i=ijk.s0;
            unsigned int j=ijk.s1;
            unsigned int k=ijk.s2;
#ifdef CONV_IDX
    unsigned int f_idx =FTNREF3D0( i, j, k, im,jm);
#else
    unsigned int f_idx = idx
#endif               
            if (i>=im || j>= jm || k>= km) {
            } else {
                if ((tc_idx != n_tchunks-1) || (NTH*tc_idx+th_id < chunk_sz)) {
                    // Compute
                    elt+=calc(mA[ f_idx ] ) ;
                }
            }
        }
        /*
	elts_th[th_id]=elt;
//        barrier(CLK_GLOBAL_MEM_FENCE);
	if (th_id==NTH-1) {
		float elts_g=0.0F;
		for (unsigned int i=0;i<NTH;i++) {
			elts_g+=elts_th[i];
		}
		mC[gr_id]=elts_g;
	}
    */

    	elts_th[th_id]=elt;
    barrier(CLK_LOCAL_MEM_FENCE);
		for (unsigned int ii=0;ii<NTH;ii++) {
			elts_g+=elts_th[ii];
		}
		mC[gr_id]=elts_g;


    }


