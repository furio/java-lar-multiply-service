// To be replaced before kernel compiling
#define MAXROW %%AROW%%
#define MAXCOL %%BCOL%%

__kernel void spmm_coo_kernel_naive(
	__global const uint * restrict ArowPtr, __global const uint * restrict Acols,
	__global const float * restrict Adata,
	__global const uint * restrict BrowPtr, __global const uint * restrict Bcols,
	__global const float * restrict Bdata,
    __global int * counter, __global float * cooVal) 
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

	float localSum = 0;

	while ((ArowCur < ArowEnd) && (BrowCur < BrowEnd)) {

		AcurIdx = Acols[ArowCur];
		BcurIdx = Bcols[BrowCur];

		if (AcurIdx == BcurIdx) {
			localSum += Adata[ArowCur] * Bdata[BrowCur];
			ArowCur++;
			BrowCur++;
		} else if ( AcurIdx < BcurIdx) {
			ArowCur++;
		} else {
			BrowCur++;
		}
	}

	if (localSum > 0) {
		int localIndex = atomic_add(counter,1);
		cooVal[localIndex*3 + 0] = (float)currRow;
		cooVal[localIndex*3 + 1] = (float)currCol;
		cooVal[localIndex*3 + 2] = localSum;
		// printf("(%d,%d)[%d]: raS: %f\n", currRow, currCol, localIndex, localSum);
	}
}

__kernel void spmm_coo_binary_kernel_naive(
	__global const uint * restrict ArowPtr, __global const uint * restrict Acols,
	__global const uint * restrict BrowPtr, __global const uint * restrict Bcols,
    __global int * counter, __global float * cooVal) 
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

	float localSum = 0;

	while ((ArowCur < ArowEnd) && (BrowCur < BrowEnd)) {

		AcurIdx = Acols[ArowCur];
		BcurIdx = Bcols[BrowCur];

		if (AcurIdx == BcurIdx) {
			localSum += 1;
			ArowCur++;
			BrowCur++;
		} else if ( AcurIdx < BcurIdx) {
			ArowCur++;
		} else {
			BrowCur++;
		}
	}

	if (localSum > 0) {
		int localIndex = atomic_add(counter,1);
		cooVal[localIndex*3 + 0] = (float)currRow;
		cooVal[localIndex*3 + 1] = (float)currCol;
		cooVal[localIndex*3 + 2] = localSum;
		// printf("(%d,%d)[%d]: raS: %f\n", currRow, currCol, localIndex, localSum);
	}
}