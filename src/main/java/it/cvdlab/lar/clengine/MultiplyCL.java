package it.cvdlab.lar.clengine;

import it.cvdlab.lar.model.CsrMatrix;

public class MultiplyCL {
	public static synchronized CsrMatrix multiply(CsrMatrix matrixA, CsrMatrix matrixB) {
		return clMultiply(matrixA, matrixB);
	}
	
	private static CsrMatrix clMultiply (CsrMatrix matrixA, CsrMatrix matrixB) {
		
		
		return null;
	}
}
