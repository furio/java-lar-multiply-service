#define MAXROW %%AROW%%
#define MAXCOL %%BCOL%%

__kernel void spmm_coo_kernel_naive(
	__global const uint * restrict ArowPtr, __global const uint * restrict Acols,
	__global const float * restrict Adata,
	__global const uint * restrict BrowPtr, __global const uint * restrict Bcols,
	__global const float * restrict Bdata,
    __global int * counter, 
    __global int * cooArr_X, __global int * cooArr_Y, __global float * cooArr_Data) 
{
	int currRow = get_global_id(0);

	if( !(currRow < MAXROW) )
	{
		return;
	}

	int ArowCur = ArowPtr[currRow];
	int ArowCur2 = ArowPtr[currRow];
	int ArowEnd = ArowPtr[currRow+1];
	
	int BrowCur = -1;
	int BrowEnd = -1;
	
	int AcurIdx = -1;
	int BcurIdx = -1;
	
	float localSum = 0;
			
	for(int currCol = 0; currCol < MAXCOL; currCol++) {
		
		ArowCur = ArowCur2;
		BrowCur = BrowPtr[currCol];
		BrowEnd = BrowPtr[currCol+1];
		
		AcurIdx = -1;
		BcurIdx = -1;
	
		localSum = 0;
	
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
			cooArr_X[localIndex] = currRow;
			cooArr_Y[localIndex] = currCol;
			cooArr_Data[localIndex] = localSum;
			// printf("(%d,%d)[%d]: raS: %f\n", currRow, currCol, localIndex, localSum);
		}
	}
}


__kernel void spmm_coo_binary_kernel_naive(
	__global const uint * restrict ArowPtr, __global const uint * restrict Acols,
	__global const uint * restrict BrowPtr, __global const uint * restrict Bcols,
    __global int * counter, 
    __global int * cooArr_X, __global int * cooArr_Y, __global float * cooArr_Data) 
{
	int currRow = get_global_id(0);

	if( !(currRow < MAXROW) )
	{
		return;
	}

	int ArowCur = ArowPtr[currRow];
	int ArowCur2 = ArowPtr[currRow];
	int ArowEnd = ArowPtr[currRow+1];
		
	int BrowCur = -1;
	int BrowEnd = -1;
	
	int AcurIdx = -1;
	int BcurIdx = -1;
	
	float localSum = 0;
			
	for(int currCol = 0; currCol < MAXCOL; currCol++) {
		
		ArowCur = ArowCur2;
		BrowCur = BrowPtr[currCol];
		BrowEnd = BrowPtr[currCol+1];
		
		AcurIdx = -1;
		BcurIdx = -1;
	
		localSum = 0;
	
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
			cooArr_X[localIndex] = currRow;
			cooArr_Y[localIndex] = currCol;
			cooArr_Data[localIndex] = localSum;
			// printf("(%d,%d)[%d]: raS: %f\n", currRow, currCol, localIndex, localSum);
		}
	}
}

/*

__kernel void spmm_coo_binary_kernel_naive(
	__global const uint * restrict ArowPtr, __global const uint * restrict Acols,
	__global const uint * restrict BrowPtr, __global const uint * restrict Bcols,
    __global int * counter, 
    __global int * cooArr_X, __global int * cooArr_Y, __global float * cooArr_Data) 
{
	int currRow = get_global_id(0);

	if( !(currRow < MAXROW) )
	{
		return;
	}

	int ArowCur = ArowPtr[currRow];
	int ArowCur2 = ArowPtr[currRow];
	int ArowEnd = ArowPtr[currRow+1];
	
	
	int AcolsLocal[128];
	for(int i = 0; (i < 128) && (ArowCur < ArowEnd); i++) {
		AcolsLocal[i] = Acols[ArowCur];
		ArowCur++;
	}
	
	// barrier(CLK_LOCAL_MEM_FENCE);
	
	int ArowEndLocal = ArowEnd - ArowCur2;
		
	int BrowCur = -1;
	int BrowEnd = -1;
	
	int AcurIdx = -1;
	int BcurIdx = -1;
	
	float localSum = 0;
			
	for(int currCol = 0; currCol < MAXCOL; currCol++) {
		
		ArowCur = 0;
		BrowCur = BrowPtr[currCol];
		BrowEnd = BrowPtr[currCol+1];
		
		AcurIdx = -1;
		BcurIdx = -1;
	
		localSum = 0;
	
		while ((ArowCur < ArowEndLocal) && (BrowCur < BrowEnd)) {
			AcurIdx = AcolsLocal[ArowCur];
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
			cooArr_X[localIndex] = currRow;
			cooArr_Y[localIndex] = currCol;
			cooArr_Data[localIndex] = localSum;
			// printf("(%d,%d)[%d]: raS: %f\n", currRow, currCol, localIndex, localSum);
		}
	}
}

*/