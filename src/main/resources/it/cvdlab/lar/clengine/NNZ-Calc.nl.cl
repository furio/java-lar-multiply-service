#define MAXROW %%AROW%%
#define MAXCOL %%BCOL%%

// #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void nnz_calc_kernel(
	__global const uint * restrict ArowPtr, __global const uint * restrict Acols,
	__global const uint * restrict BrowPtr, __global const uint * restrict Bcols,
    __global int * counter) 
{
	int currRow = get_global_id(0);
	int currCol = get_global_id(1);

//	if( !((currRow < MAXROW) && (currCol < MAXCOL)) ) {
//		return;
//	}

	int ArowCur = ArowPtr[currRow];
	int ArowEnd = ArowPtr[currRow+1];
	
	int BrowCur = BrowPtr[currCol];
	int BrowEnd = BrowPtr[currCol+1];
	
	int AcurIdx = -1;
	int BcurIdx = -1;
	
	// printf("(%d,%d): raS: %d raE: %d rbS: %d rbE: %d\n", currRow, currCol, ArowCur, ArowEnd, BrowCur, BrowEnd);
	
	bool haveNNZ = false;

	while ((ArowCur < ArowEnd) && (BrowCur < BrowEnd)) {

		AcurIdx = Acols[ArowCur];
		BcurIdx = Bcols[BrowCur];

		if (AcurIdx == BcurIdx) {
			haveNNZ = true;
			break;
		} else if ( AcurIdx < BcurIdx) {
			ArowCur++;
		} else {
			BrowCur++;
		}
	}
	
	if (haveNNZ == true) {
		atomic_add(counter,1);
	}
}