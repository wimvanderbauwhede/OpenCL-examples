// This is a reusable function to evaluate the effect of the loop ordering
#define LOOP_ORDER 1

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
#elif LOOP_ORDER == 3 
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
