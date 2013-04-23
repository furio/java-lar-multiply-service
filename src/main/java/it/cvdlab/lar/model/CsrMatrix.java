package it.cvdlab.lar.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.codehaus.jackson.annotate.JsonIgnore;
import org.codehaus.jackson.annotate.JsonProperty;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;

public class CsrMatrix {
	@JsonProperty("ROW")
	private List<Integer> rowptr;
	@JsonProperty("COL")
	private List<Integer> coldata;
	@JsonProperty("DATA")
	private List<Float> data;
	@JsonProperty("ROWCOUNT")
	private int rowshape;
	@JsonProperty("COLCOUNT")
	private int colshape;
	
	public CsrMatrix(List<Integer> rowPtr, List<Integer> colData, List<Float> data, int rowshape, int colshape) {
		this.rowptr = ImmutableList.copyOf( rowPtr );
		this.coldata = ImmutableList.copyOf( colData );
		this.data = ImmutableList.copyOf( data );
		this.rowshape = rowshape;
		this.colshape = colshape;
	}
	
	public CsrMatrix(int rowPtr[], int[] colData, float[] data, int rowshape, int colshape) {
		this( Ints.asList(rowPtr), Ints.asList(colData), Floats.asList( data ), rowshape, colshape);
	}
	
	public CsrMatrix(int rowPtr[], int[] colData, int rowshape, int colshape) {
		this( rowPtr, colData, binarydataInit(colData.length, 1F) , rowshape, colshape);
	}
	
	private static float[] binarydataInit(int length, float initValue) {
		float[] arr = new float[length];
		Arrays.fill(arr, initValue);
		return arr;
	}	
	
	@JsonIgnore
	public boolean isBinary() {
		for(Float currData: data) {
			if ((currData != 0F) && (currData != 1F)) {
				return false;
			}
		}
		
		return true;
	}
	
	@JsonIgnore
	public List<List<Float>> toDense() {
		List<List<Float>> returnMatrix = new ArrayList<List<Float>>( this.getRowshape() );
		
		for(int i = 0; i < (this.getRowptr().size() - 1); i++) {
			List<Float> curRow = new ArrayList<Float>(Collections.nCopies(this.getColshape(), 0F));
			for(int k = this.getRowptr().get(i); k < this.getRowptr().get(i+1); k++) {
				curRow.set( this.getColdata().get(k), this.getData().get(k) );
			}
			returnMatrix.add( new ArrayList<Float>(curRow) );
		}
		
		return returnMatrix;
	}
	
	@JsonIgnore
	public CsrMatrix transpose() {		
		// lookup
		int m = this.getRowshape();
		int n = this.getColshape();
		int base = 0;

		// NNZ elements
		int nnz = this.getRowptr().get(m) - base;

		// New arrays
		int[] newPtr = new int[n + 1];
		int[] newCol = new int[nnz];
		float[] newData = new float[nnz];
		// Create and initialize to 0
		int[] count_nnz = new int[n];

		// Reused index
		int i = 0;

		// Count nnz per column
		for(i = 0; i < nnz; i++) {
			count_nnz[(this.getColdata().get(i) - base)]++;
		}
		
		transposeHelper(count_nnz, n, newPtr);
		
		// Copia TrowPtr in modo tale che count_nnz[i] == location in Tind, Tval
		for(i = 0; i < n; i++) {
			count_nnz[i] = newPtr[i];
		}
		
		// Copia i valori in posizione
		for(i = 0; i < m; i++) {
			int k;
			for (k = (this.getRowptr().get(i) - base); k < (this.getRowptr().get(i+1) - base); k++ ) {
				int j = this.getColdata().get(k) - base;
				int l = count_nnz[j];

				newCol[l] = i;
				newData[l] = this.getData().get(k);
				count_nnz[j]++;
			}
		}		
		
		return new CsrMatrix(newPtr, newCol, newData, n, m);
	}
	
	private static void transposeHelper(int[] input, int maxN, int[] output) {
		if (maxN == 0) {
			return;
		}

		output[0] = 0;
		for (int i = 1; i <= maxN; i++) {
			output[i] = output[i - 1] + input[i - 1];
		}
	}
	
	@JsonIgnore
	public CsrMatrix multiply(CsrMatrix matrix) throws Exception {
		if (this.getColshape() != matrix.getRowshape()) {
			throw new Exception("Current matrix columns are different from argument matrix rows");
		}

		CsrMatrix argMatrix = matrix.transpose();
	    float[] denseResult = new float[(this.getRowshape() * matrix.getColshape())];

	    for (int i = 0; i < this.getRowshape(); i++) {
			for (int j = 0; j < argMatrix.getRowshape(); j++) {

				int ArowCur = this.getRowptr().get(i),
					ArowEnd = this.getRowptr().get(i + 1),
					curPosA = ArowCur;

				int BrowCur = argMatrix.getRowptr().get(j),
					BrowEnd = argMatrix.getRowptr().get(j + 1),
					curPosB = BrowCur;

				int AcurIdx = this.getColdata().get(ArowCur),
					BcurIdx = argMatrix.getColdata().get(BrowCur);

	            float localSum = 0F;

	            while ((curPosA < ArowEnd) && (curPosB < BrowEnd)) {
					AcurIdx = this.getColdata().get(curPosA);
					BcurIdx = argMatrix.getColdata().get(curPosB);

					if (AcurIdx == BcurIdx) {
						localSum += this.getData().get(curPosA) * argMatrix.getData().get(curPosB);
						curPosA++;
						curPosB++;
					} else if (AcurIdx < BcurIdx) {
						curPosA++;
					} else {
						curPosB++;
					}
				}

				denseResult[i*matrix.getColshape() + j] = localSum;
			}
		}

		return fromFlattenArray(denseResult, matrix.getColshape());		
	}	
	
	// For compatibility with JS code
	@JsonIgnore
	public int getRowCount() {
		return getRowshape();
	}
	@JsonIgnore
	public int getColCount() {
		return getColshape();
	}
	// ------------------------------
	
	public List<Integer> getRowptr() {
		return rowptr;
	}
	public void setRowptr(List<Integer> rowptr) {
		this.rowptr = rowptr;
	}
	public List<Integer> getColdata() {
		return coldata;
	}
	public void setColdata(List<Integer> coldata) {
		this.coldata = coldata;
	}
	public List<Float> getData() {
		return data;
	}
	public void setData(List<Float> data) {
		this.data = data;
	}
	public int getRowshape() {
		return rowshape;
	}
	public void setRowshape(int rowshape) {
		this.rowshape = rowshape;
	}
	public int getColshape() {
		return colshape;
	}
	public void setColshape(int colshape) {
		this.colshape = colshape;
	}	
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((coldata == null) ? 0 : coldata.hashCode());
		result = prime * result + (int) (colshape ^ (colshape >>> 32));
		result = prime * result + ((data == null) ? 0 : data.hashCode());
		result = prime * result + ((rowptr == null) ? 0 : rowptr.hashCode());
		result = prime * result + (int) (rowshape ^ (rowshape >>> 32));
		return result;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		CsrMatrix other = (CsrMatrix) obj;
		if (coldata == null) {
			if (other.coldata != null)
				return false;
		} else if (!coldata.equals(other.coldata))
			return false;
		if (colshape != other.colshape)
			return false;
		if (data == null) {
			if (other.data != null)
				return false;
		} else if (!data.equals(other.data))
			return false;
		if (rowptr == null) {
			if (other.rowptr != null)
				return false;
		} else if (!rowptr.equals(other.rowptr))
			return false;
		if (rowshape != other.rowshape)
			return false;
		return true;
	}

	@Override
	public String toString() {
		return "CsrMatrix [rowptr=" + rowptr + ", coldata=" + coldata
				+ ", data=" + data + ", rowshape=" + rowshape + ", colshape="
				+ colshape + "]";
	}
	
	public static CsrMatrix fromFlattenArray(int[] input, int columns) {	
		float[] fInput = new float[input.length];
		for(int i = 0; i < input.length; i++) {
			fInput[i] = input[i];
		}
		
		return fromFlattenArray(fInput,columns);
	}
	
	public static CsrMatrix fromFlattenArray(float[] input, int columns) {		
		List<Integer> rowPtr = Lists.newArrayList();
		List<Integer> cols = Lists.newArrayList();
		List<Float> data = Lists.newArrayList();
		int nnz = 0;
		int colIdx = 0;
		int rowCount = 0;
		int prevRow = -1;
		
		for (int i = 0; i < input.length; i++, colIdx++) {
			if (prevRow != rowCount) {
				rowPtr.add( nnz );
				prevRow = rowCount;
			}

			if ( input[i] != 0F ) {
				cols.add( colIdx );
				data.add( input[i] );
				nnz += 1;
			}

			if ((colIdx+1) == columns) {
				colIdx = -1;
				rowCount += 1;
			}
		}
		
		rowPtr.add( data.size() );
		
		return new CsrMatrix(rowPtr, cols, data, rowPtr.size() - 1, columns);
	}
}