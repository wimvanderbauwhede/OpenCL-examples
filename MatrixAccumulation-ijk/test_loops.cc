#include <iostream>

struct int4 {
    int s0;
    int s1;
    int s2;
    int s3;
};

#include "calc_loop_iters.h"
#include "calc_array_index.h"

inline int calc(int x) {
    int v= x;
    for (unsigned int n= 0; n< (2<<SELECT);n++) {
        v+=2;
        v/=2;
    }
    return v;
}

int main() {

    const unsigned int im = 150;
    const unsigned int jm = 150;
    const unsigned int km = 90;
    const unsigned int ijkm = im*jm*km;
    const unsigned int nunits = NGROUPS;
    int mC[NGROUPS];
    for (unsigned int i=0;i<nunits;i++) {
        mC[i]=0;
    }
    int* mA = new int[ijkm];
 int ref = 0;
    for (unsigned int i=0;i<ijkm;i++) {
        ref+=1;
//         std::cout << i << "\n";
        mA[i]=1;
    }
    std::cout << "IJKM: "<< ijkm<<"\n";
    for (unsigned int get_global_id=0; get_global_id<nunits*NTH; get_global_id++) {
        unsigned int get_local_id = get_global_id % NTH;
        unsigned int get_group_id = get_global_id/NTH;
        unsigned int get_num_groups=nunits;
#ifdef VV    
        std::cout <<  "GROUP ID: " <<get_group_id << "\n";
        std::cout <<  "THREAD  ID: " <<get_local_id << "\n";
#endif    
        //    unsigned int gl_id = get_global_id;
        unsigned int g_id = get_group_id;
        unsigned int nunits=get_num_groups;
        //    __local int elts_th[NTH];
        // calculate the local range from idx and the full range :
        unsigned int ijkm = im*jm*km;
        unsigned int chunk_sz = ijkm/nunits;//*NTH);
        unsigned int chunk_idx = chunk_sz*g_id;
        // Now, this will be too small in some cases
        unsigned int th_id = get_local_id;
        // WRONG! Instead, correct the last group!
        if (g_id == nunits-1) {
            if (ijkm % nunits !=0) {
#ifdef VV    
                std::cout << "Corrected last work group ("<< g_id<<") chunk with " <<ijkm % nunits << " for thread " << th_id<<"\n";
#endif
                chunk_sz+=ijkm % nunits;
            }
        }
#ifdef VV    
        std::cout << "CHUNKSZ: " << chunk_sz << "\n";
#endif
        // starting index
        unsigned int n_tchunks = chunk_sz/NTH;
        // WRONG: instead, correct the last thread
        n_tchunks += (chunk_sz % NTH == 0) ? 0 : 1;

#ifdef VV    
        std::cout << "#CHUNKS: " << n_tchunks << " of size " << NTH << "\n";
#endif    
        // now we can iterate over this range in chunks of nthreads
        int elt=0;
        for (unsigned int tc_idx = 0 ; tc_idx< n_tchunks; tc_idx++) {
            // the actual index in the overall array must be
            unsigned int a_idx = chunk_idx + tc_idx*NTH + th_id;
            //            std::cout << a_idx;
            // From this we can now calc i,j,k
            int4 ijk = calc_loop_iters(a_idx,im, jm, km,0,0,0) ;
            unsigned int i=ijk.s0;
            unsigned int j=ijk.s1;
            unsigned int k=ijk.s2;
            unsigned int f_idx =  FTNREF3D0( i, j, k, im,jm);
            if (i>=im || j>= jm || k>= km) {
                std::cout << "INDEX a_idx TOO LARGE: "<< a_idx<<" ("<< i<< "," << j << ","  <<k <<") => " << f_idx << " for tc_idx="<<tc_idx<<" , th_id="<<th_id<<" , g_id="<<g_id<<"\n";
            } else {
                if ((tc_idx != n_tchunks-1) || (NTH*tc_idx+th_id < chunk_sz)) {


                    // Compute
                    elt+=mA[ f_idx ]  ;
                }
            }
        }
        mC[g_id]+=elt;

    }
    int ct=0;
    for (unsigned int i=0;i<nunits;i++) {
        //         std::cout << i << "\n";
        ct+=mC[i];
    }
    std::cout << ct << " <> " << ref <<"\n";// (ijkm-1)*ijkm/2 << "\n";
    std::cout << ((ct - ref) ? "NOK" : "OK") <<" " << (ct - ref) << "\n";// (ijkm-1)*ijkm/2 << "\n";
    delete[] mA;
    return(1);
}
