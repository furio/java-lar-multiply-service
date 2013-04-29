#define MAXROW %%AROW%%
#define MAXCOL %%BCOL%%

bool JCmult(int colsize, int rowsize, int *Aj, int *Bi) {
	for(int j = 0; j < rowsize;j++) {
		for(int k = 0;k < colsize;k++) {
			if(Aj[j] == Bi[k]) {
				return true;
			}
		}
	}
	
	return false;
}


__kernel void findJCkernel(__global int *Aptr, __global int *Aj, __global int *Bptr, __global int *Bi, __global int *Cjc)
{
	int i;
	int tid = get_global_id(0); // threadIdx.x + blockIdx.x * blockDim.x;
	for(i = 0;i < MAXCOL; i++) {
		int cbeg = Bptr[i];
		int cend = Bptr[i+1];
		if(tid < MAXROW) {
			int beg = Aptr[tid];
			int end = Aptr[tid + 1];
			if ( JCmult(cend - cbeg, end - beg, Aj+beg, Bi+cbeg) ) {
				atomicAdd(Cjc+i+1,1);
			}
		}
	}
}