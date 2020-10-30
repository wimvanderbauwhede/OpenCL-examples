/* This kernel is inspired by the odd/even mutual recursion, but generalised over a modulo of N
 * the algorithm is trivial, the aim is to investigate hand-over of work across forever-running threads.
 * 
 * As expected, this does _not_ work, the system hangs. 
 */
__kernel void modulo (
		__global unsigned int *number,		
		__global unsigned int *active_thread,
		__global unsigned int *res,
		const unsigned int mod
) {

	unsigned int g_id=get_global_id(0);			
	unsigned int at=active_thread[0];
	unsigned int n=number[0];
	 
	if (at==g_id) {
		if ( n==1 ) {
			res[0] = at+1;
		} else if ( n==0 ) {
			res[0] = -1;	
		} else {
			active_thread[0] = (at+1) % mod;
			number[0]=n-1;
		}					
	}		
}
