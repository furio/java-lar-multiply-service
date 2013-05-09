package it.cvdlab.lar.clengine;

import it.cvdlab.lar.model.CsrMatrix;

import java.util.Map;

import org.bridj.Pointer;

import com.google.common.collect.Maps;
import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLMem;

final class MultiplyCLStatus {
	// qui il context
	private CLContext context = null;
	
	//
	private Map<String,CLMem> buffersRelease = Maps.newHashMap();
	private Map<String,Pointer<Integer>> pointersIntegerRelease = Maps.newHashMap();
	private Map<String,Pointer<Float>> pointersFloatRelease = Maps.newHashMap();
	// =======================
	
	// qui il valore di nnz
	private int nnz;
	
	// qui matrix a,b
	private CsrMatrix a,b;
	// =======================
	
	// il method free
    public void free() {
		clearAllocatedCLObjects();
		clearAllocatedPTRObjects();
		if (context != null) {
			context.release();
		}
		context = null;
		//
		a = null;
		b = null;
    }
    
    public void releaseSingleCL(String key) {
    	if (this.buffersRelease.containsKey(key)) {
        	this.buffersRelease.get(key).release();
        	this.buffersRelease.remove(key);
    	}
    }

    public void releaseSinglePTR(String key) {
    	if (this.pointersIntegerRelease.containsKey(key)) {
        	this.pointersIntegerRelease.get(key).release();
        	this.pointersIntegerRelease.remove(key);    		
    	} else if (this.pointersFloatRelease.containsKey(key)) {
        	this.pointersFloatRelease.get(key).release();
        	this.pointersFloatRelease.remove(key);     		
    	}
    }
    
	public void clearAllocatedCLObjects() {
		System.err.println("Clearing CLMEM");
		for(String buffObject: this.buffersRelease.keySet()) {
			this.buffersRelease.get(buffObject).release();
		}
		this.buffersRelease.clear();
	}
	
	public void clearAllocatedPTRObjects() {
		System.err.println("Clearing POINTERS");
		for(String buffObject: this.pointersIntegerRelease.keySet()) {
			this.pointersIntegerRelease.get(buffObject).release();
		}
		this.pointersIntegerRelease.clear();
		//
		for(String buffObject: this.pointersFloatRelease.keySet()) {
			this.pointersFloatRelease.get(buffObject).release();
		}
		this.pointersFloatRelease.clear();		
	}

	@SuppressWarnings("unused")
	private Map<String, Pointer<Integer>> getPointersIntegerRelease() {
		return pointersIntegerRelease;
	}
	
	public Pointer<Integer> getPointerInteger(String key) {
		return pointersIntegerRelease.get(key);
	}
	
	public Pointer<Integer> setPointerInteger(String key, Pointer<Integer> value) {
		return pointersIntegerRelease.put(key,value);
	}

	@SuppressWarnings("unused")
	private Map<String, Pointer<Float>> getPointersFloatRelease() {
		return pointersFloatRelease;
	}
	
	public Pointer<Float> getPointerFloat(String key) {
		return pointersFloatRelease.get(key);
	}
	
	public Pointer<Float> setPointerFloat(String key, Pointer<Float> value) {
		return pointersFloatRelease.put(key,value);
	}	

	public int getNnz() {
		return nnz;
	}

	public void setNnz(int nnz) {
		this.nnz = nnz;
	}

	public CsrMatrix getMatrixA() {
		return a;
	}

	public void setMatrixA(CsrMatrix a) {
		this.a = a;
	}

	public CsrMatrix getMatrixB() {
		return b;
	}

	public void setMatrixB(CsrMatrix b) {
		this.b = b;
	}

	public Map<String, CLMem> getBuffersRelease() {
		return buffersRelease;
	}
	
	@SuppressWarnings("unchecked")
	private <T> CLBuffer<T> getBuffer(String key) {
		return ( CLBuffer<T> )buffersRelease.get(key);
	}
	
	public CLBuffer<Integer> getBufferInteger(String key) {
		return getBuffer(key);
	}
	
	public CLBuffer<Float> getBufferFloat(String key) {
		return getBuffer(key);
	}
	
	@SuppressWarnings("unchecked")
	private <T> CLBuffer<T> setBuffer(String key, CLBuffer<T> value) {
		return ( CLBuffer<T> )buffersRelease.put(key,value);
	}	
	
	public CLBuffer<Integer> setBufferInteger(String key, CLBuffer<Integer> value) {
		return setBuffer(key,value);
	}
	
	public CLBuffer<Float> setBufferFloat(String key, CLBuffer<Float> value) {
		return setBuffer(key,value);
	}	

	public CLContext getContext() {
		return context;
	}

	public void setContext(CLContext context) {
		this.context = context;
	}
}
