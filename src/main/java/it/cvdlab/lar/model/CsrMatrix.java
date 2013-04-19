package it.cvdlab.lar.model;

import java.lang.reflect.Array;
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
		this( Ints.asList(rowPtr), Ints.asList(colData), Floats.asList( data ), colshape, colshape);
	}
	
	public CsrMatrix(int rowPtr[], int[] colData, int rowshape, int colshape) {
		this( rowPtr, colData, binarydataInit(colData.length, 1F) , rowshape, colshape);
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
			returnMatrix.set(i, new ArrayList<Float>(curRow) );
		}
		
		return returnMatrix;
	}
	
	@JsonIgnore
	public CsrMatrix transpose() {
		// TODO: refactor from JS
		/*

	// private function
	var f_transposeEnum = function(inputArray, maxN, outputArray) {
		if (maxN === 0) {
			return;
		}

		outputArray[0] = 0;
		for (var i = 1; i <= maxN; i++) {
			outputArray[i] = outputArray[i - 1] + inputArray[i - 1];
		}
	};

	// lookup
	var m = this.getRowCount();
	var n = this.getColCount();
	var base = this.baseIndex;

	// NNZ elements
	var nnz = this.getRowPointer()[m] - base;

	// New arrays
	var newPtr = new Array(n + 1);
	var newCol = new Array(nnz);
	var newData = new Array(nnz);
	// Create and initialize to 0
	var count_nnz = newFilledArray(n, 0);

	// Reused index
	var i = 0;

	// Count nnz per column
	for(i = 0; i < nnz; i++) {
		count_nnz[(this.getColumnIndices()[i] - base)]++;
	}

	// Create the new rowPtr
	f_transposeEnum(count_nnz, n, newPtr);

	// Copia TrowPtr in moda tale che count_nnz[i] == location in Tind, Tval
	for(i = 0; i < n; i++) {
		count_nnz[i] = newPtr[i];
	}

	// Copia i valori in posizione
	for(i = 0; i < m; i++) {
		var k;
		for (k = (this.getRowPointer()[i] - base); k < (this.getRowPointer()[i+1] - base); k++ ) {
			var j = this.getColumnIndices()[k] - base;
			var l = count_nnz[j];

			newCol[l] = i;
			newData[l] = this.getData()[k];
			count_nnz[j]++;
		}
	}

	return new csr_matrix({"numrows": n, "numcols": m, "rowptr": newPtr, "colindices": newCol, "data": newData});

		 */
		return null;
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

			if ( input[i] != 0 ) {
				cols.add( colIdx );
				data.add( input[i] );
				nnz += 1;
			}

			if ((colIdx+1) == columns) {
				colIdx = -1;
				rowCount += 1;
			}
		}		
		
		return new CsrMatrix(rowPtr, cols, data, rowPtr.size() - 1, columns);
	}
	
	private static float[] binarydataInit(int length, float initValue) {
		float[] arr = new float[length];
		Arrays.fill(arr, initValue);
		return arr;
	}
}
