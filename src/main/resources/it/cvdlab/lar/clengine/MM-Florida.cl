#define MAXROW %%AROW%%
#define MAXCOL %%BCOL%%

void mult(int szrow, int szcol, int *Aj, int *Bi, float *Av, float *Bv,
		int rowid, int colid, int *count, Point *dP) {
	float sum = 0.0f;
	for(int j = 0; j < szrow; j++) {
		for(int k = 0; k < szcol; k++) {
			if(Aj[j] == Bi[k]) {
				sum = sum + Av[j] * Bv[k];
			}
		}
	}

	if(sum) {
		int index = atomicAdd(count,1);
		(dP + index)->x = rowid+1;
		(dP + index)->y = colid+1;
		(dP + index)->val = sum;
	}
}

__kernel void mmkernel(int *Aptr, int *Acol, int *Bptr, int *Brow,
		float *Aval, float* Bval, Point *dP, int *count) {
	int i;
	int tid = get_global_id(0); // threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < MAXROW) {
		int beg = Aptr[tid];
		int end = Aptr[tid+1];
		for(i = 0;i < MAXCOL; i++) {
			int cbeg = Bptr[i];
			int cend = Bptr[i+1];
			mult(end-beg, cend-cbeg, Acol+beg, Brow+cbeg, Aval+beg, Bval+cbeg, tid, i, count, dP);
		}
	}
}