package it.cvdlab.lar.clengine;


import it.cvdlab.lar.clengine.utils.PonterUtils;
import it.cvdlab.lar.clengine.utils.SizeEstimator;
import it.cvdlab.lar.model.CsrMatrix;

import java.io.IOException;
import java.nio.ByteOrder;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.bridj.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLException;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.util.IOUtils;

final class MultiplyCLCached {
	// Logger
	private static final Logger logger = LoggerFactory.getLogger(MultiplyCLCached.class);
	
	// Buffer identifier (TODO)
	
	static CsrMatrix multiply(CsrMatrix matrixA, CsrMatrix matrixB, boolean forceCOO) {
		// Dense result
		long denseResult = matrixA.getRowCount();
		denseResult *= (long)matrixB.getColCount();
		
		// Init cache object
		MultiplyCLStatus clCache = new MultiplyCLStatus();
		clCache.setMatrixA(matrixA);
		clCache.setMatrixB(matrixB);
		
		// Context
		clCache.setContext( KernelConfig.createContext() );
		
		if (clCache.getContext() == null) {
			clCache.free();
			return null;    	
        }		
		
		try {
			clCache.setNnz( clCalcNNZ( clCache ) );
		} catch (Exception e) {
			logger.error(e.toString());
			return null; 
		}
		
		System.err.println("NNZ Res: " + clCache.getNnz());
		
		CsrMatrix resultMatrix = null;
		if ( forceCOO || CLEngineConfig.isUSECOO() || (denseResult > ( clCache.getNnz() * CLEngineConfig.getNNZ_WEIGHT() )) ) {
			System.err.println("COO Way");
			resultMatrix = clMultiplyCOO(clCache);
		} else {
			System.err.println("Dense Way");
			resultMatrix = clMultiply(clCache);
		}
		
		if ( CLEngineConfig.isFORCE_GC() ) {
			System.gc();
			System.gc();
		}
		
		return resultMatrix;
	}
	
	private static CsrMatrix clMultiply(MultiplyCLStatus clCache) {
		CsrMatrix matrixA = clCache.getMatrixA();
		CsrMatrix matrixBToTranspose = clCache.getMatrixB();

		// WorkGroupSize
		long maxWorkGroupSize = Long.MAX_VALUE;
		for(CLDevice currDev: clCache.getContext().getDevices() ) {
			maxWorkGroupSize = Math.min(maxWorkGroupSize, currDev.getMaxWorkGroupSize());
		}
		
        CLQueue queue = clCache.getContext().createDefaultQueue();
        ByteOrder byteOrder = clCache.getContext().getByteOrder();
        
        CsrMatrix matrixB = matrixBToTranspose.transpose();
        boolean isBinary = matrixA.isBinary() && matrixB.isBinary();
        
        long startTime = System.currentTimeMillis();
        
        // Native memory
        // Pointer<Float> matA_data = null, matB_data = null;
        // Pointer<Integer> matA_rowptr, matA_colindices, matB_rowptr, matB_colindices;
        
        // Allocate
        if (!isBinary) {
            clCache.setPointerFloat( "matA_data", Pointer.allocateFloats(matrixA.getData().size()).order(byteOrder) );
            clCache.setPointerFloat( "matB_data", Pointer.allocateFloats(matrixB.getData().size()).order(byteOrder) );
        }

        if (!isBinary) {
        	PonterUtils.copyToPointer(matrixA.getData(), clCache.getPointerFloat( "matA_data" ) );
        	PonterUtils.copyToPointer(matrixB.getData(), clCache.getPointerFloat( "matB_data" ) );
        }
        
        
        // CLBuffers
        //  CLBuffer<Integer> cl_matA_rowptr = null, cl_matA_colindices = null, cl_matB_rowptr = null, cl_matB_colindices = null;
        // CLBuffer<Float> cl_matA_data = null, cl_matB_data = null;
        // CLBuffer<Float> cl_output_data = null;
        
        try {
            if (!isBinary) {
            	clCache.setBufferFloat( "cl_matA_data", clCache.getContext().createBuffer(Usage.Input, clCache.getPointerFloat( "matA_data" ), CLEngineConfig.isUSE_DEVICE_MEM() ) );
            	clCache.setBufferFloat( "cl_matB_data", clCache.getContext().createBuffer(Usage.Input, clCache.getPointerFloat( "matB_data" ), CLEngineConfig.isUSE_DEVICE_MEM() ) );
            }
            
            // Output buffer
            clCache.setBufferFloat( "cl_output_data", clCache.getContext().createFloatBuffer(Usage.Output, matrixA.getRowCount()*matrixBToTranspose.getColCount() ) );
        } catch (CLException e) {
			queue.flush();
			queue.release();
			clCache.free();
			
			System.err.println(e.toString());
			return null;        	
        }


        // Read the program sources and compile them :
        String kernelSource = null;
		try {
			kernelSource = IOUtils.readText(MultiplyCLCached.class.getResource(KernelConfig.KERNEL_DENSE()));
		} catch (IOException e) {
			queue.flush();
			queue.release();
			clCache.free();
			
			System.err.println(e.toString());
			return null;
		}
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_ROW, Integer.toString( matrixA.getRowCount() ) );
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_COL, Integer.toString( matrixB.getRowCount() ) );
        
    	// System.out.println(kernelSource);
        
        CLProgram program = clCache.getContext().createProgram(kernelSource);

        // Get and call the kernel :
        CLKernel multiplyMatrixKernel = null;
        if (!isBinary) {
        	multiplyMatrixKernel = program.createKernel(KernelConfig.KERNEL_DENSE_FUN_FULL);
        	multiplyMatrixKernel.setArgs(
        			clCache.getBufferInteger("cl_matA_rowptr"),
        			clCache.getBufferInteger("cl_matA_colindices"),
        			clCache.getBufferFloat("cl_matA_data"),
        			clCache.getBufferInteger("cl_matB_rowptr"),
        			clCache.getBufferInteger("cl_matB_colindices"),
        			clCache.getBufferFloat("cl_matB_data"),
        			clCache.getBufferFloat("cl_output_data")
        			);
        } else {
        	multiplyMatrixKernel = program.createKernel(KernelConfig.KERNEL_DENSE_FUN_SHORT);
        	multiplyMatrixKernel.setArgs(
        			clCache.getBufferInteger("cl_matA_rowptr"),
        			clCache.getBufferInteger("cl_matA_colindices"),
        			clCache.getBufferInteger("cl_matB_rowptr"),
        			clCache.getBufferInteger("cl_matB_colindices"),
        			clCache.getBufferFloat("cl_output_data")
        			);
        }
        
        int[] wgSize;
        int[] locSize;
        
        if (CLEngineConfig.isIMPL_LOCAL()) {
        	wgSize = new int[]{matrixA.getRowCount(), matrixB.getRowCount()};
        	locSize = null;
        } else {
    		try {
    			List<int[]> niceSizes = SizeEstimator.getGoodSizes(matrixA.getRowCount(), matrixB.getRowCount(), (int) maxWorkGroupSize);
    			wgSize = niceSizes.get(0);
    			locSize = niceSizes.get(1);
    		} catch (Exception e) {
    			queue.flush();
    			queue.release();
    			multiplyMatrixKernel.release();
    			program.release();
    			clCache.free();
    			
    			System.err.println(e.toString());
    			return null;
    		}        	
        }

        // queue.finish();
        CLEvent addEvt = null;
        if (CLEngineConfig.isIMPL_LOCAL()) {
        	addEvt = multiplyMatrixKernel.enqueueNDRange(queue, wgSize);
        } else {
        	addEvt = multiplyMatrixKernel.enqueueNDRange(queue, wgSize, locSize);
        }
        
        clCache.setPointerFloat( "matrixDataOut", clCache.getBufferFloat("cl_output_data").read(queue, addEvt) );
        // Pointer<Float> matrixDataOut = Pointer.allocateFloats(matrixA.getRowCount()*matrixBToTranspose.getColCount()).order(byteOrder);
        // cl_output_data.read(queue, matrixDataOut, true, addEvt);
        
        List<Float> listMatrixOut = PonterUtils.copyFromPointerFloat( clCache.getPointerFloat( "matrixDataOut") );
        
        addEvt.release();
        queue.flush();
        queue.release();
		multiplyMatrixKernel.release();
		program.release();
		clCache.free();
        
		// System.out.println(listMatrixOut);
		System.err.println("Calculated in: " + (System.currentTimeMillis() - startTime) + " millis");
		
		return CsrMatrix.fromFlattenArray(ArrayUtils.toPrimitive( listMatrixOut.toArray(new Float[0]) ), matrixBToTranspose.getColCount());
	}
	
	
	private static CsrMatrix clMultiplyCOO(MultiplyCLStatus clCache) {
		CsrMatrix matrixA = clCache.getMatrixA();
		CsrMatrix matrixBToTranspose = clCache.getMatrixB();
		int nnzCount = clCache.getNnz();

		
		// WorkGroupSize
		long maxWorkGroupSize = Long.MAX_VALUE;
		for(CLDevice currDev: clCache.getContext().getDevices() ) {
			maxWorkGroupSize = Math.min(maxWorkGroupSize, currDev.getMaxWorkGroupSize());
		}
		
        CLQueue queue = clCache.getContext().createDefaultQueue();
        ByteOrder byteOrder = clCache.getContext().getByteOrder();
        
        CsrMatrix matrixB = matrixBToTranspose.transpose();
        boolean isBinary = matrixA.isBinary() && matrixB.isBinary();
        
        long startTime = System.currentTimeMillis();
        
        // Native memory
        // Pointer<Float> matA_data = null, matB_data = null;
        // Pointer<Integer> counter, matA_rowptr, matA_colindices, matB_rowptr, matB_colindices;
        
        // Allocate
        clCache.setPointerInteger("counter", Pointer.allocateInt().order(byteOrder) );
        clCache.getPointerInteger("counter").set(0);
        if (!isBinary) {
            clCache.setPointerFloat( "matA_data", Pointer.allocateFloats(matrixA.getData().size()).order(byteOrder) );
            clCache.setPointerFloat( "matB_data", Pointer.allocateFloats(matrixB.getData().size()).order(byteOrder) );
        }

        if (!isBinary) {
        	PonterUtils.copyToPointer(matrixA.getData(), clCache.getPointerFloat( "matA_data" ) );
        	PonterUtils.copyToPointer(matrixB.getData(), clCache.getPointerFloat( "matB_data" ) );
        }
        
        
        // CLBuffers
        // CLBuffer<Integer> cl_counter = null, cl_matA_rowptr = null, cl_matA_colindices = null, cl_matB_rowptr = null, cl_matB_colindices = null;
        // CLBuffer<Float> cl_matA_data = null, cl_matB_data = null;
        // CLBuffer<Float> cl_output_data = null;
        // CLBuffer<Integer> cl_output_data_x = null, cl_output_data_y = null;
        // CLBuffer<Float> cl_output_data_val = null;
        
        try {
        	// Always use device mem for the counter
        	clCache.setBufferInteger( "cl_counter", clCache.getContext().createBuffer(Usage.InputOutput, clCache.getPointerInteger("counter")) );
            if (!isBinary) {
            	clCache.setBufferFloat( "cl_matA_data", clCache.getContext().createBuffer(Usage.Input, clCache.getPointerFloat( "matA_data" ), CLEngineConfig.isUSE_DEVICE_MEM() ) );
            	clCache.setBufferFloat( "cl_matB_data", clCache.getContext().createBuffer(Usage.Input, clCache.getPointerFloat( "matB_data" ), CLEngineConfig.isUSE_DEVICE_MEM() ) );
            }
            
            // Output buffer
            clCache.setBufferInteger( "cl_output_data_x", clCache.getContext().createIntBuffer(Usage.Output, nnzCount) );
            clCache.setBufferInteger( "cl_output_data_y", clCache.getContext().createIntBuffer(Usage.Output, nnzCount) );
            clCache.setBufferFloat( "cl_output_data_val", clCache.getContext().createFloatBuffer(Usage.Output, nnzCount) );
        } catch (CLException e) {
        	queue.flush();
			queue.release();
			clCache.free();
			
			System.err.println(e.toString());
			return null;    	
        }


        // Read the program sources and compile them :
        String kernelSource = null;
		try {
			kernelSource = IOUtils.readText(MultiplyCLCached.class.getResource(KernelConfig.KERNEL_COO()));
		} catch (IOException e) {
			queue.flush();
			queue.release();
			clCache.free();
			
			System.err.println(e.toString());
			return null;
		}
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_ROW, Integer.toString( matrixA.getRowCount() ) );
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_COL, Integer.toString( matrixB.getRowCount() ) );
        
    	// System.out.println(kernelSource);
        
        CLProgram program = clCache.getContext().createProgram(kernelSource);

        // Get and call the kernel :
        CLKernel multiplyMatrixKernel = null;
        if (!isBinary) {
        	multiplyMatrixKernel = program.createKernel(KernelConfig.KERNEL_COO_FUN_FULL);
        	multiplyMatrixKernel.setArgs(
        			clCache.getBufferInteger("cl_matA_rowptr"),
        			clCache.getBufferInteger("cl_matA_colindices"),
        			clCache.getBufferFloat("cl_matA_data"),
        			clCache.getBufferInteger("cl_matB_rowptr"),
        			clCache.getBufferInteger("cl_matB_colindices"),
        			clCache.getBufferFloat("cl_matB_data"),
        			clCache.getBufferInteger("cl_counter"),
        			clCache.getBufferInteger("cl_output_data_x"),
        			clCache.getBufferInteger("cl_output_data_y"),
        			clCache.getBufferFloat("cl_output_data_val")
        			);
        } else {
        	multiplyMatrixKernel = program.createKernel(KernelConfig.KERNEL_COO_FUN_SHORT);
        	multiplyMatrixKernel.setArgs(
        			clCache.getBufferInteger("cl_matA_rowptr"),
       				clCache.getBufferInteger("cl_matA_colindices"),
        			clCache.getBufferInteger("cl_matB_rowptr"),
        			clCache.getBufferInteger("cl_matB_colindices"),
        			clCache.getBufferInteger("cl_counter"),
        			clCache.getBufferInteger("cl_output_data_x"),
        			clCache.getBufferInteger("cl_output_data_y"),
        			clCache.getBufferFloat("cl_output_data_val")
        			);
        }
        
        int[] wgSize;
        int[] locSize;
        
        if (CLEngineConfig.isIMPL_LOCAL()) {
        	wgSize = new int[]{matrixA.getRowCount(), matrixB.getRowCount()};
        	locSize = null;
        } else {
    		try {
    			List<int[]> niceSizes = SizeEstimator.getGoodSizes(matrixA.getRowCount(), matrixB.getRowCount(), (int) maxWorkGroupSize);
    			wgSize = niceSizes.get(0);
    			locSize = niceSizes.get(1);
    		} catch (Exception e) {
    			queue.flush();
    			queue.release();
    			multiplyMatrixKernel.release();
    			program.release();
    			clCache.free();
    			
    			System.err.println(e.toString());
    			return null;
    		}        	
        }

        // queue.finish();
        CLEvent addEvt = null;
        if (CLEngineConfig.isIMPL_LOCAL()) {
        	addEvt = multiplyMatrixKernel.enqueueNDRange(queue, wgSize);
        } else {
        	addEvt = multiplyMatrixKernel.enqueueNDRange(queue, wgSize, locSize);
        }
        
        clCache.setPointerInteger( "matrixDataOut_x", clCache.getBufferInteger("cl_output_data_x").read(queue, addEvt) );
        clCache.setPointerInteger( "matrixDataOut_y", clCache.getBufferInteger("cl_output_data_y").read(queue, addEvt) );
        clCache.setPointerFloat( "matrixDataOut_val", clCache.getBufferFloat("cl_output_data_val").read(queue, addEvt) );
        // Pointer<Float> matrixDataOut = Pointer.allocateFloats(matrixA.getRowCount()*matrixBToTranspose.getColCount()).order(byteOrder);
        // cl_output_data.read(queue, matrixDataOut, true, addEvt);
        
        List<Integer> listMatrixOut_x = PonterUtils.copyFromPointerInteger( clCache.getPointerInteger( "matrixDataOut_x") );
        List<Integer> listMatrixOut_y = PonterUtils.copyFromPointerInteger( clCache.getPointerInteger( "matrixDataOut_y") );
        List<Float> listMatrixOut_val = PonterUtils.copyFromPointerFloat( clCache.getPointerFloat( "matrixDataOut_val") );
        
        addEvt.release();
        queue.flush();
        queue.release();
		multiplyMatrixKernel.release();
		program.release();
		clCache.free();
		
		System.err.println("Calculated in: " + (System.currentTimeMillis() - startTime) + " millis");
		
		return CsrMatrix.fromCOOArray(listMatrixOut_x, listMatrixOut_y, listMatrixOut_val, matrixA.getRowshape(), matrixBToTranspose.getColshape());
	}
	
	private static int clCalcNNZ(MultiplyCLStatus clCache) {		
		// WorkGroupSize
		long maxWorkGroupSize = Long.MAX_VALUE;
		for(CLDevice currDev: clCache.getContext().getDevices() ) {
			maxWorkGroupSize = Math.min(maxWorkGroupSize, currDev.getMaxWorkGroupSize());
		}
		
        CLQueue queue = clCache.getContext().createDefaultQueue();
        ByteOrder byteOrder = clCache.getContext().getByteOrder();
        
        CsrMatrix matrixA = clCache.getMatrixA();
        CsrMatrix matrixB = clCache.getMatrixB().transpose();
        
        long startTime = System.currentTimeMillis();
        
        // Native memory
        // counter, matA_rowptr, matA_colindices, matB_rowptr, matB_colindices;
        
        // Allocate
        clCache.setPointerInteger("counter", Pointer.allocateInt().order(byteOrder) );
        clCache.getPointerInteger("counter").set(0);
        
        clCache.setPointerInteger( "matA_rowptr", Pointer.allocateInts(matrixA.getRowptr().size()).order(byteOrder) );
        clCache.setPointerInteger( "matA_colindices", Pointer.allocateInts(matrixA.getColdata().size()).order(byteOrder) );
        clCache.setPointerInteger( "matB_rowptr", Pointer.allocateInts(matrixB.getRowptr().size()).order(byteOrder) );
        clCache.setPointerInteger( "matB_colindices", Pointer.allocateInts(matrixB.getColdata().size()).order(byteOrder) );
        
        PonterUtils.copyToPointer(matrixA.getRowptr(), clCache.getPointerInteger("matA_rowptr") );
        PonterUtils.copyToPointer(matrixA.getColdata(), clCache.getPointerInteger("matA_colindices") );
        PonterUtils.copyToPointer(matrixB.getRowptr(), clCache.getPointerInteger("matB_rowptr") );
        PonterUtils.copyToPointer(matrixB.getColdata(), clCache.getPointerInteger("matB_colindices") );
        
        // CLBuffers
        try {
        	// Always use device mem for the counter
        	clCache.setBufferInteger( "cl_counter", clCache.getContext().createBuffer(Usage.InputOutput, clCache.getPointerInteger("counter")) );
        	clCache.setBufferInteger( "cl_matA_rowptr", clCache.getContext().createBuffer(Usage.Input, clCache.getPointerInteger("matA_rowptr"), CLEngineConfig.isUSE_DEVICE_MEM() ) );
        	clCache.setBufferInteger( "cl_matA_colindices", clCache.getContext().createBuffer(Usage.Input, clCache.getPointerInteger("matA_colindices"), CLEngineConfig.isUSE_DEVICE_MEM() ) );
        	clCache.setBufferInteger( "cl_matB_rowptr", clCache.getContext().createBuffer(Usage.Input, clCache.getPointerInteger("matB_rowptr"), CLEngineConfig.isUSE_DEVICE_MEM() ) );
            clCache.setBufferInteger( "cl_matB_colindices", clCache.getContext().createBuffer(Usage.Input, clCache.getPointerInteger("matB_colindices"), CLEngineConfig.isUSE_DEVICE_MEM() ));
        } catch (CLException e) {
        	queue.flush();
			queue.release();
			clCache.free();
			
			System.err.println(e.toString());
			return -1;    	
        }


        // Read the program sources and compile them :
        String kernelSource = null;
		try {
			kernelSource = IOUtils.readText(MultiplyCLCached.class.getResource(KernelConfig.KERNEL_NNZ()));
		} catch (IOException e) {
			queue.flush();
			queue.release();
			clCache.free();
			
			System.err.println(e.toString());
			return -1;
		}
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_ROW, Integer.toString( matrixA.getRowCount() ) );
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_COL, Integer.toString( matrixB.getRowCount() ) );
        
    	// System.out.println(kernelSource);
        
        CLProgram program = clCache.getContext().createProgram(kernelSource);

        // Get and call the kernel :
        CLKernel multiplyMatrixKernel = null;

       	multiplyMatrixKernel = program.createKernel(KernelConfig.KERNEL_NNZ_FUN);
       	multiplyMatrixKernel.setArgs(
       				clCache.getBufferInteger("cl_matA_rowptr"),
       				clCache.getBufferInteger("cl_matA_colindices"),
        			clCache.getBufferInteger("cl_matB_rowptr"),
        			clCache.getBufferInteger("cl_matB_colindices"),
        			clCache.getBufferInteger("cl_counter") );
        
        int[] wgSize;
        int[] locSize;
        
        if (CLEngineConfig.isIMPL_LOCAL()) {
        	wgSize = new int[]{matrixA.getRowCount(), matrixB.getRowCount()};
        	locSize = null;
        } else {
    		try {
    			List<int[]> niceSizes = SizeEstimator.getGoodSizes(matrixA.getRowCount(), matrixB.getRowCount(), (int) maxWorkGroupSize);
    			wgSize = niceSizes.get(0);
    			locSize = niceSizes.get(1);
    		} catch (Exception e) {
    			queue.flush();
    			queue.release();
    			multiplyMatrixKernel.release();
    			program.release();
    			clCache.free();
    			
    			System.err.println(e.toString());
    			return -1;
    		}        	
        }

        // queue.finish();
        CLEvent addEvt = null;
        if (CLEngineConfig.isIMPL_LOCAL()) {
        	addEvt = multiplyMatrixKernel.enqueueNDRange(queue, wgSize);
        } else {
        	addEvt = multiplyMatrixKernel.enqueueNDRange(queue, wgSize, locSize);
        } 
       
        clCache.setPointerInteger("counter", clCache.getBufferInteger("cl_counter").read(queue, addEvt) );
        // Pointer<Float> matrixDataOut = Pointer.allocateFloats(matrixA.getRowCount()*matrixBToTranspose.getColCount()).order(byteOrder);
        // cl_output_data.read(queue, matrixDataOut, true, addEvt);
        
        int resultCount = clCache.getPointerInteger("counter").get();
		
        addEvt.release();
        queue.flush();
        queue.release();
		multiplyMatrixKernel.release();
		program.release();
		
		clCache.releaseSingleCL("cl_counter");
		clCache.releaseSinglePTR("counter");
		
		System.err.println("Calculated in: " + (System.currentTimeMillis() - startTime) + " millis");
        
        return resultCount;
	}
}
