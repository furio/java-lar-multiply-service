package it.cvdlab.lar.clengine;

import it.cvdlab.lar.model.CsrMatrix;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

final class CLInputSizeEstimator {
	@SuppressWarnings("unused")
	private static final Logger logger = LoggerFactory.getLogger(CLInputSizeEstimator.class);
	private static final int BIT_PER_BYTES = Byte.SIZE;
	private static final int COO_TRIPLET = 3;
	
	private int genericInputSize(CsrMatrix a, CsrMatrix b) {
		int firstMatrix = a.getRowPointer().size() + a.getColdata().size();
		int secondMatrix = b.getRowPointer().size() + b.getColdata().size();
		
		int inputSize = (firstMatrix + secondMatrix) * (Integer.SIZE / BIT_PER_BYTES);
		
		return inputSize;
	}
	
	public int denseSize(CsrMatrix a, CsrMatrix b) {
		int firstMatrix = a.getNonZeroElementsCount();
		int secondMatrix = b.getNonZeroElementsCount();
		
		int additionalInput = (firstMatrix + secondMatrix) * (Float.SIZE / BIT_PER_BYTES);
		
		return additionalInput + denseSizeBinary(a,b);
	}
	
	public int denseSizeBinary(CsrMatrix a, CsrMatrix b) { 
		int outputSize = a.getRowCount() * b.getColCount() * (Float.SIZE / BIT_PER_BYTES);
		
		return genericInputSize(a,b) + outputSize;
	}
	
	public int cooSize(CsrMatrix a, CsrMatrix b) {
		int firstMatrix = a.getNonZeroElementsCount();
		int secondMatrix = b.getNonZeroElementsCount();
		
		int additionalInput = (firstMatrix + secondMatrix) * (Float.SIZE / BIT_PER_BYTES);
		
		return additionalInput + cooSize(a,b);
	}
	
	public int cooSizeBinary(CsrMatrix a, CsrMatrix b, int nnzElements) { 
		int outputSize = nnzElements * COO_TRIPLET * (Float.SIZE / BIT_PER_BYTES);
		
		return genericInputSize(a,b) + outputSize;
	}	
}