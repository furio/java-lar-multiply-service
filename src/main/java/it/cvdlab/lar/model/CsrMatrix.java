package it.cvdlab.lar.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.codehaus.jackson.annotate.JsonIgnore;
import org.codehaus.jackson.annotate.JsonProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
	@JsonIgnore
	private static final int BASEINDEX = 0;
	@JsonIgnore
	public static boolean USE_SPARSE_MULTIPLY = false;	
	@JsonIgnore
	private static final Logger logger = LoggerFactory.getLogger(CsrMatrix.class);
	
	public CsrMatrix() {}
	
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
	
	@JsonIgnore
	public boolean isBinary() {
		for(Float currData: this.getData()) {
			if (currData != 1F) {
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
		int base = BASEINDEX;

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

		if ( USE_SPARSE_MULTIPLY ) {
			return this.sparseMultiply(matrix);
		} else {
			return this.denseMultiply(matrix);
		}
		
	}
	
	@JsonIgnore
	private CsrMatrix sparseMultiply(CsrMatrix matrix) {
		int i,j,k,l,count,countJD;
		int[] JD = new int[matrix.getColCount()];
		
		int[] tmp_rowptr, tmp_coldata;
		float[] tmp_data;
		int tmp_rowcount = this.getRowCount(),
				tmp_colcount = matrix.getColCount();
		
		// Init JD
		for (i=0; i < tmp_colcount; ++i) {
			JD[i] = -1;
		}
		
		// Init rowPtr
		tmp_rowptr = new int[tmp_rowcount + 1];
		
		// Calculate rowPtr
		for(i = 0; i < tmp_rowcount; ++i) {
			count = 0; 

			for (k = this.getRowPointer().get(i); k < this.getRowPointer().get(i + 1); ++k) {
				for (j = matrix.getRowPointer().get( this.getColdata().get(k) ); j < matrix.getRowPointer().get( this.getColdata().get(k) + 1 ); ++j) {
					for (l=0; l<count; l++) {
						if ( JD[l] == matrix.getColdata().get(j) ) {
							break;
						}
					}

					if ( l == count ) {
						JD[count] = matrix.getColdata().get(j);
						count++;
					}
				}
			}

			tmp_rowptr[i+1] = count;
			for (j=0; j < count; ++j) {
				JD[j] = -1;
			}
		}
		
		// Finally set
		for ( i = 0; i < tmp_rowcount; ++i) { 
			tmp_rowptr[i+1] += tmp_rowptr[i];
		}
		
		// Init tmpColData
		tmp_coldata = new int[tmp_rowptr[tmp_rowcount]];
		
		// These fors have a bug, at the first step inverts some column indexes
		// need to be debugged
		for (i=0; i < tmp_rowcount; ++i) {
			countJD = 0;
			count = tmp_rowptr[i];
			for ( k = this.getRowPointer().get(i); k < this.getRowPointer().get(i + 1); ++k) {
				for ( j = matrix.getRowPointer().get( this.getColdata().get(k) ); j < matrix.getRowPointer().get( this.getColdata().get(k) + 1 ); ++j) {
					for ( l=0; l<countJD; l++) {
						if ( JD[l] == matrix.getColdata().get(j) ) {
							break;
						}
					}

					if ( l == countJD ) {
						tmp_coldata[count] = matrix.getColdata().get(j);
						JD[countJD] = matrix.getColdata().get(j);
						count++;
						countJD++;
					}
				}
			}

			for ( j=0; j < countJD; ++j) {
				JD[j] = -1;
			}
		}
		
		// Init data
		tmp_data = new float[tmp_rowptr[tmp_rowcount]];
		
		for (i=0; i < tmp_rowcount; ++i) {
			for (j = tmp_rowptr[i]; j < tmp_rowptr[i + 1]; ++j) {
				tmp_data[j] = 0;
				for ( k = this.getRowPointer().get(i); k < this.getRowPointer().get(i+1); ++k) {
					for ( l = matrix.getRowPointer().get( this.getColdata().get(k) ); l < matrix.getRowPointer().get( this.getColdata().get(k) + 1 ); l++) {
						if ( matrix.getColdata().get(l) == tmp_coldata[j] ) {
							tmp_data[j] += this.getData().get(k) * matrix.getData().get(l);
						} // end if
					} // end for l
				} // end for k
			} // end for j
		} // end for i
  		
		
		return new CsrMatrix(tmp_rowptr, tmp_coldata, tmp_data, tmp_rowcount, tmp_colcount);
	}

	@JsonIgnore
	private CsrMatrix denseMultiply(CsrMatrix matrix) {
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
	
	@JsonIgnore
	public CsrMatrix getRowPiece(int i, int j, int length) throws Exception {
		if ((i < 1) || (j < 1) || (length < 1)) {
			throw new Exception("Parameters must be > 1");
		}

		if (this.getRowptr().size() < 2) {
			throw new Exception("Matrix is not valid");
		}

		if (i > this.getRowCount()) {
			throw new Exception("Row index cannot be > RowCount");
		}

		if (j > this.getColCount()) {
			throw new Exception("Col index cannot be > ColCount");
		}

		if (length > (this.getColCount() - j + 1)) {
			logger.debug("Warning: out of the bound of the matrix, padded with 0");
		}

		List<Integer> tmp_col = Lists.newArrayList();
		List<Float> tmp_data = Lists.newArrayList();

		if (this.getNonZeroElementsCount() > 0) {
			// Matrix is valid so getRowPointer ever has an element in position [i-1]
			int ptr = this.getRowPointer().get(i - 1);
			int end_ptr = this.getRowPointer().get(i);

			int column = this.getColumnIndices().get(ptr) + 1;

			while ((ptr < end_ptr) && (column < (j + length))) {
				
				if (column > (j - 1)) {
					tmp_col.add(column - j);
					tmp_data.add(this.getData().get(ptr));
				}

				ptr++;
				column = this.getColumnIndices().get(ptr) + 1;
			}
		}
		
		return new CsrMatrix(Ints.asList(new int[]{0,length}), tmp_col, tmp_data, 1, length);
	}
	
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
	public synchronized List<Float> getData() {
		// Lazy init
		if (this.data == null) {
			this.data = ImmutableList.copyOf( Floats.asList( binarydataInit(this.getColdata().size(), 1F) ) );
		}
		
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
	
	// For compatibility with JS code
	@JsonIgnore
	public int getRowCount() {
		return getRowshape();
	}
	@JsonIgnore
	public int getColCount() {
		return getColshape();
	}
	@JsonIgnore
	public int getNonZeroElementsCount() {
		return this.getData().size();
	}
	@JsonIgnore
	public List<Integer> getRowPointer() {
		return this.getRowptr();
	}
	@JsonIgnore
	public List<Integer> getColumnIndices() {
		return this.getColdata();
	}
	// ------------------------------
	
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
	
	private static float[] binarydataInit(int length, float initValue) {
		float[] arr = new float[length];
		Arrays.fill(arr, initValue);
		return arr;
	}
}
