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

import com.google.common.collect.Lists;
import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLException;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.util.IOUtils;

public final class MultiplyCL {
	// Logger
	private static final Logger logger = LoggerFactory.getLogger(MultiplyCL.class);
	
	public static synchronized CsrMatrix multiply(CsrMatrix matrixA, CsrMatrix matrixB, boolean forceCOO) {
		System.err.println("===");
		System.err.println("A Res: " + matrixA.getRowCount() + "x" + matrixA.getColCount() + " NNz: " + matrixA.getColdata().size());
		System.err.println("B Res: " + matrixB.getRowCount() + "x" + matrixB.getColCount() + " NNz: " + matrixB.getColdata().size());
		long denseResult = matrixA.getRowCount();
		denseResult *= (long)matrixB.getColCount();
		System.err.println("Dim Res: " + denseResult);	
		
		// Js-like computation
		// TODO: There is a bug in which if A*B and B has 1 col OpenCL kernel compute strange stuff
		if (CLEngineConfig.isNO_OPENCL() || (matrixB.getColCount() == 1)) {
			System.err.println("== JS Multiply ==");
			return jsMultiply(matrixA, matrixB);
		}
		
		// Use the cached shared object way
		if (CLEngineConfig.isSHARED_CL()) {
			System.err.println("== Cached CL ==");
			return MultiplyCLCached.multiply(matrixA, matrixB, forceCOO);
		}
		
		// Go through OpenCL
		// First calculate NNZ elements!! This is necessary!
		int nnzCount = -1;
		
		try {
			nnzCount = matrixA.nnzMultiplyCount(matrixB);
//			nnzCount = clCalcNNZ(matrixA,matrixB);
//			System.out.println("NNZCount: " + nnzCount );
		} catch (Exception e) {
			logger.error(e.toString());
			return null; 
		}
		
		System.err.println("NNZ Res: " + nnzCount);
		
		CsrMatrix resultMatrix = null;
		if ( forceCOO || CLEngineConfig.isUSECOO() || (denseResult > ( nnzCount * CLEngineConfig.getNNZ_WEIGHT() )) ) {
			System.err.println("COO Way");
			// resultMatrix = clMultiplyCOO(matrixA, matrixB, nnzCount);
			resultMatrix = clMultiplyCOO_FLAT(matrixA, matrixB, nnzCount);
		} else {
			System.err.println("Dense Way");
			resultMatrix = clMultiply(matrixA, matrixB);
		}
		
		if ( CLEngineConfig.isFORCE_GC() ) {
			System.gc();
			System.gc();
		}
		
		return resultMatrix;
	}
	private static CsrMatrix jsMultiply(CsrMatrix matrixA, CsrMatrix matrixBToTranspose) {
		try {
			return matrixA.multiply(matrixBToTranspose);
		} catch (Exception e) {
			logger.error(e.toString());
			return null;
		}
	}
	
	private static CsrMatrix clMultiply(CsrMatrix matrixA, CsrMatrix matrixBToTranspose) {
		// Lista di CL buffer da deallocare
		List<CLMem> buffersRelease = Lists.newArrayList();
		@SuppressWarnings("rawtypes")
		List<Pointer> pointersRelease = Lists.newArrayList();
		
		long startTime = System.currentTimeMillis();
		
		CLContext context = KernelConfig.createContext();
		
		if (context == null) {
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);

			return null;    	
        }
		
		// WorkGroupSize
		long maxWorkGroupSize = Long.MAX_VALUE;
		for(CLDevice currDev: context.getDevices() ) {
			maxWorkGroupSize = Math.min(maxWorkGroupSize, currDev.getMaxWorkGroupSize());
		}
		
        CLQueue queue = context.createDefaultQueue();
        ByteOrder byteOrder = context.getByteOrder();
        
        CsrMatrix matrixB = matrixBToTranspose.transpose();
        boolean isBinary = matrixA.isBinary() && matrixB.isBinary();
        
        // Native memory
        Pointer<Float> matA_data = null, matB_data = null;
        Pointer<Integer> matA_rowptr, matA_colindices, matB_rowptr, matB_colindices;
        
        // Allocate
        matA_rowptr = Pointer.allocateInts(matrixA.getRowptr().size()).order(byteOrder);
        pointersRelease.add(matA_rowptr);
        matA_colindices = Pointer.allocateInts(matrixA.getColdata().size()).order(byteOrder);
        pointersRelease.add(matA_colindices);
        matB_rowptr = Pointer.allocateInts(matrixB.getRowptr().size()).order(byteOrder);
        pointersRelease.add(matB_rowptr);
        matB_colindices = Pointer.allocateInts(matrixB.getColdata().size()).order(byteOrder);
        pointersRelease.add(matB_colindices);
        if (!isBinary) {
        	matA_data = Pointer.allocateFloats(matrixA.getData().size()).order(byteOrder);
        	pointersRelease.add(matA_data);
        	matB_data = Pointer.allocateFloats(matrixB.getData().size()).order(byteOrder);
        	pointersRelease.add(matB_data);
        }
        
        PonterUtils.copyToPointer(matrixA.getRowptr(), matA_rowptr);
        PonterUtils.copyToPointer(matrixA.getColdata(), matA_colindices);
        PonterUtils.copyToPointer(matrixB.getRowptr(), matB_rowptr);
        PonterUtils.copyToPointer(matrixB.getColdata(), matB_colindices);
        if (!isBinary) {
        	PonterUtils.copyToPointer(matrixA.getData(), matA_data);
        	PonterUtils.copyToPointer(matrixB.getData(), matB_data);
        }
        
        
        // CLBuffers
        CLBuffer<Integer> cl_matA_rowptr = null, cl_matA_colindices = null, cl_matB_rowptr = null, cl_matB_colindices = null;
        CLBuffer<Float> cl_matA_data = null, cl_matB_data = null;
        CLBuffer<Float> cl_output_data = null;
        
        try {
            cl_matA_rowptr = context.createBuffer(Usage.Input, matA_rowptr, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matA_rowptr);
            cl_matA_colindices = context.createBuffer(Usage.Input, matA_colindices, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matA_colindices);
            cl_matB_rowptr = context.createBuffer(Usage.Input, matB_rowptr, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matB_rowptr);
            cl_matB_colindices = context.createBuffer(Usage.Input, matB_colindices, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matB_colindices);
            if (!isBinary) {
            	cl_matA_data = context.createBuffer(Usage.Input, matA_data, CLEngineConfig.isUSE_DEVICE_MEM());
            	buffersRelease.add(cl_matA_data);
            	cl_matB_data = context.createBuffer(Usage.Input, matB_data, CLEngineConfig.isUSE_DEVICE_MEM());
            	buffersRelease.add(cl_matB_data);
            }
            
            // Output buffer
            cl_output_data = context.createFloatBuffer(Usage.Output, matrixA.getRowCount()*matrixBToTranspose.getColCount());
            buffersRelease.add(cl_output_data);
        } catch (CLException e) {
			queue.flush();
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);
			context.release();
			
			System.err.println(e.toString());
			return null;        	
        }


        // Read the program sources and compile them :
        String kernelSource = null;
		try {
			kernelSource = IOUtils.readText(MultiplyCL.class.getResource( KernelConfig.KERNEL_DENSE() ));
		} catch (IOException e) {
			queue.flush();
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);
			context.release();
			
			System.err.println(e.toString());
			return null;
		}
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_ROW, Integer.toString( matrixA.getRowCount() ) );
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_COL, Integer.toString( matrixB.getRowCount() ) );
        
    	// System.out.println(kernelSource);
        
        CLProgram program = context.createProgram(kernelSource);

        // Get and call the kernel :
        CLKernel multiplyMatrixKernel = null;
        if (!isBinary) {
        	multiplyMatrixKernel = program.createKernel( KernelConfig.KERNEL_DENSE_FUN_FULL );
        	multiplyMatrixKernel.setArgs(cl_matA_rowptr,
        			cl_matA_colindices,
        			cl_matA_data,
        			cl_matB_rowptr,
        			cl_matB_colindices,
        			cl_matB_data,
        			cl_output_data);
        } else {
        	multiplyMatrixKernel = program.createKernel( KernelConfig.KERNEL_DENSE_FUN_SHORT );
        	multiplyMatrixKernel.setArgs(cl_matA_rowptr,
        			cl_matA_colindices,
        			cl_matB_rowptr,
        			cl_matB_colindices,
        			cl_output_data);
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
    			clearAllocatedCLObjects(buffersRelease);
    			clearAllocatedPTRObjects(pointersRelease);
    			context.release();
    			
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
        
        Pointer<Float> matrixDataOut = cl_output_data.read(queue, addEvt);
        pointersRelease.add(matrixDataOut);
        // Pointer<Float> matrixDataOut = Pointer.allocateFloats(matrixA.getRowCount()*matrixBToTranspose.getColCount()).order(byteOrder);
        // cl_output_data.read(queue, matrixDataOut, true, addEvt);
        
        List<Float> listMatrixOut = PonterUtils.copyFromPointerFloat(matrixDataOut);
        
        addEvt.release();
        queue.flush();
        queue.release();
		multiplyMatrixKernel.release();
		program.release();
		clearAllocatedCLObjects(buffersRelease);
		clearAllocatedPTRObjects(pointersRelease);
		context.release();
        
		// System.out.println(listMatrixOut);
		System.err.println("Calculated in: " + (System.currentTimeMillis() - startTime) + " millis");
		
		return CsrMatrix.fromFlattenArray(ArrayUtils.toPrimitive( listMatrixOut.toArray(new Float[0]) ), matrixBToTranspose.getColCount());
	}
	
	
	private static CsrMatrix clMultiplyCOO(CsrMatrix matrixA, CsrMatrix matrixBToTranspose, int nnzCount) {
		// Lista di CL buffer da deallocare
		List<CLMem> buffersRelease = Lists.newArrayList();
		@SuppressWarnings("rawtypes")
		List<Pointer> pointersRelease = Lists.newArrayList();
		
		long startTime = System.currentTimeMillis();
		
		//
		CLContext context = KernelConfig.createContext();
		
		if (context == null) {
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);

			return null;    	
        }
		// Context

		
		// WorkGroupSize
		long maxWorkGroupSize = Long.MAX_VALUE;
		for(CLDevice currDev: context.getDevices() ) {
			maxWorkGroupSize = Math.min(maxWorkGroupSize, currDev.getMaxWorkGroupSize());
		}
		
        CLQueue queue = context.createDefaultQueue();
        ByteOrder byteOrder = context.getByteOrder();
        
        CsrMatrix matrixB = matrixBToTranspose.transpose();
        boolean isBinary = matrixA.isBinary() && matrixB.isBinary();
        
        // Native memory
        Pointer<Float> matA_data = null, matB_data = null;
        Pointer<Integer> counter, matA_rowptr, matA_colindices, matB_rowptr, matB_colindices;
        
        // Allocate
        counter = Pointer.allocateInt().order(byteOrder);
        counter.set(0);
        pointersRelease.add(counter);
        matA_rowptr = Pointer.allocateInts(matrixA.getRowptr().size()).order(byteOrder);
        pointersRelease.add(matA_rowptr);
        matA_colindices = Pointer.allocateInts(matrixA.getColdata().size()).order(byteOrder);
        pointersRelease.add(matA_colindices);
        matB_rowptr = Pointer.allocateInts(matrixB.getRowptr().size()).order(byteOrder);
        pointersRelease.add(matB_rowptr);
        matB_colindices = Pointer.allocateInts(matrixB.getColdata().size()).order(byteOrder);
        pointersRelease.add(matB_colindices);
        if (!isBinary) {
        	matA_data = Pointer.allocateFloats(matrixA.getData().size()).order(byteOrder);
        	pointersRelease.add(matA_data);
        	matB_data = Pointer.allocateFloats(matrixB.getData().size()).order(byteOrder);
        	pointersRelease.add(matB_data);
        }
        
        PonterUtils.copyToPointer(matrixA.getRowptr(), matA_rowptr);
        PonterUtils.copyToPointer(matrixA.getColdata(), matA_colindices);
        PonterUtils.copyToPointer(matrixB.getRowptr(), matB_rowptr);
        PonterUtils.copyToPointer(matrixB.getColdata(), matB_colindices);
        if (!isBinary) {
        	PonterUtils.copyToPointer(matrixA.getData(), matA_data);
        	PonterUtils.copyToPointer(matrixB.getData(), matB_data);
        }
        
        
        // CLBuffers
        CLBuffer<Integer> cl_counter = null, cl_matA_rowptr = null, cl_matA_colindices = null, cl_matB_rowptr = null, cl_matB_colindices = null;
        CLBuffer<Float> cl_matA_data = null, cl_matB_data = null;
        CLBuffer<Integer> cl_output_data_x = null, cl_output_data_y = null;
        CLBuffer<Float> cl_output_data_val = null;
        
        try {
        	// Always use device mem for the counter
        	cl_counter = context.createBuffer(Usage.InputOutput, counter);
        	buffersRelease.add(cl_counter);
            cl_matA_rowptr = context.createBuffer(Usage.Input, matA_rowptr, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matA_rowptr);
            cl_matA_colindices = context.createBuffer(Usage.Input, matA_colindices, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matA_colindices);
            cl_matB_rowptr = context.createBuffer(Usage.Input, matB_rowptr, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matB_rowptr);
            cl_matB_colindices = context.createBuffer(Usage.Input, matB_colindices, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matB_colindices);
            if (!isBinary) {
            	cl_matA_data = context.createBuffer(Usage.Input, matA_data, CLEngineConfig.isUSE_DEVICE_MEM());
            	buffersRelease.add(cl_matA_data);
            	cl_matB_data = context.createBuffer(Usage.Input, matB_data, CLEngineConfig.isUSE_DEVICE_MEM());
            	buffersRelease.add(cl_matB_data);
            }
            
            // Output buffer
            cl_output_data_x = context.createIntBuffer(Usage.Output, nnzCount);
            buffersRelease.add(cl_output_data_x);
            cl_output_data_y = context.createIntBuffer(Usage.Output, nnzCount);
            buffersRelease.add(cl_output_data_y);
            cl_output_data_val = context.createFloatBuffer(Usage.Output, nnzCount);
            buffersRelease.add(cl_output_data_val);            
        } catch (CLException e) {
        	queue.flush();
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);
			context.release();
			
			System.err.println(e.toString());
			return null;    	
        }


        // Read the program sources and compile them :
        String kernelSource = null;
		try {
			kernelSource = IOUtils.readText(MultiplyCL.class.getResource( KernelConfig.KERNEL_COO() ));
		} catch (IOException e) {
			queue.flush();
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);
			context.release();
			
			System.err.println(e.toString());
			return null;
		}
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_ROW, Integer.toString( matrixA.getRowCount() ) );
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_COL, Integer.toString( matrixB.getRowCount() ) );
        
    	// System.out.println(kernelSource);
        
        CLProgram program = context.createProgram(kernelSource);

        // Get and call the kernel :
        CLKernel multiplyMatrixKernel = null;
        if (!isBinary) {
        	multiplyMatrixKernel = program.createKernel(KernelConfig.KERNEL_COO_FUN_FULL);
        	multiplyMatrixKernel.setArgs(cl_matA_rowptr,
        			cl_matA_colindices,
        			cl_matA_data,
        			cl_matB_rowptr,
        			cl_matB_colindices,
        			cl_matB_data,
        			cl_counter,
        			cl_output_data_x,
        			cl_output_data_y,
        			cl_output_data_val);
        } else {
        	multiplyMatrixKernel = program.createKernel(KernelConfig.KERNEL_COO_FUN_SHORT);
        	multiplyMatrixKernel.setArgs(cl_matA_rowptr,
        			cl_matA_colindices,
        			cl_matB_rowptr,
        			cl_matB_colindices,
        			cl_counter,
        			cl_output_data_x,
        			cl_output_data_y,
        			cl_output_data_val);
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
    			clearAllocatedCLObjects(buffersRelease);
    			clearAllocatedPTRObjects(pointersRelease);
    			context.release();
    			
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
       
        Pointer<Integer> matrixDataOut_x = cl_output_data_x.read(queue, addEvt);
        System.err.println("Calculated in: " + (System.currentTimeMillis() - startTime) + " millis");
        pointersRelease.add(matrixDataOut_x);
        Pointer<Integer> matrixDataOut_y = cl_output_data_y.read(queue);
        pointersRelease.add(matrixDataOut_y);
        Pointer<Float> matrixDataOut_val = cl_output_data_val.read(queue);
        pointersRelease.add(matrixDataOut_val);
        // Pointer<Float> matrixDataOut = Pointer.allocateFloats(matrixA.getRowCount()*matrixBToTranspose.getColCount()).order(byteOrder);
        // cl_output_data.read(queue, matrixDataOut, true, addEvt);
        
        List<Integer> listMatrixOut_x = PonterUtils.copyFromPointerInteger(matrixDataOut_x);
        List<Integer> listMatrixOut_y = PonterUtils.copyFromPointerInteger(matrixDataOut_y);
        List<Float> listMatrixOut_val = PonterUtils.copyFromPointerFloat(matrixDataOut_val);
		
        addEvt.release();
        queue.flush();
        queue.release();
		multiplyMatrixKernel.release();
		program.release();
		clearAllocatedCLObjects(buffersRelease);
		clearAllocatedPTRObjects(pointersRelease);
		context.release();
		
//		System.out.println(listMatrixOut_x);
//		System.out.println(listMatrixOut_y);
//		System.out.println(listMatrixOut_val);
		System.err.println("Calculated in: " + (System.currentTimeMillis() - startTime) + " millis");
		
		return CsrMatrix.fromCOOArray(listMatrixOut_x, listMatrixOut_y, listMatrixOut_val, matrixA.getRowshape(), matrixBToTranspose.getColshape());
	}
	
	private static CsrMatrix clMultiplyCOO_FLAT(CsrMatrix matrixA, CsrMatrix matrixBToTranspose, int nnzCount) {
		// Lista di CL buffer da deallocare
		List<CLMem> buffersRelease = Lists.newArrayList();
		@SuppressWarnings("rawtypes")
		List<Pointer> pointersRelease = Lists.newArrayList();
		
		long startTime = System.currentTimeMillis();
		
		//
		CLContext context = KernelConfig.createContext();
		
		if (context == null) {
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);

			return null;    	
        }
		// Context

		
		// WorkGroupSize
		long maxWorkGroupSize = Long.MAX_VALUE;
		for(CLDevice currDev: context.getDevices() ) {
			maxWorkGroupSize = Math.min(maxWorkGroupSize, currDev.getMaxWorkGroupSize());
		}
		
        CLQueue queue = context.createDefaultQueue();
        ByteOrder byteOrder = context.getByteOrder();
        
        CsrMatrix matrixB = matrixBToTranspose.transpose();
        boolean isBinary = matrixA.isBinary() && matrixB.isBinary();
        // System.err.println("IsBinary: " + isBinary);
        
        // Native memory
        Pointer<Float> matA_data = null, matB_data = null;
        Pointer<Integer> counter, matA_rowptr, matA_colindices, matB_rowptr, matB_colindices;
        
        // Allocate
        counter = Pointer.allocateInt().order(byteOrder);
        counter.set(0);
        pointersRelease.add(counter);
        matA_rowptr = Pointer.allocateInts(matrixA.getRowptr().size()).order(byteOrder);
        pointersRelease.add(matA_rowptr);
        matA_colindices = Pointer.allocateInts(matrixA.getColdata().size()).order(byteOrder);
        pointersRelease.add(matA_colindices);
        matB_rowptr = Pointer.allocateInts(matrixB.getRowptr().size()).order(byteOrder);
        pointersRelease.add(matB_rowptr);
        matB_colindices = Pointer.allocateInts(matrixB.getColdata().size()).order(byteOrder);
        pointersRelease.add(matB_colindices);
        if (!isBinary) {
        	matA_data = Pointer.allocateFloats(matrixA.getData().size()).order(byteOrder);
        	pointersRelease.add(matA_data);
        	matB_data = Pointer.allocateFloats(matrixB.getData().size()).order(byteOrder);
        	pointersRelease.add(matB_data);
        }
        
        PonterUtils.copyToPointer(matrixA.getRowptr(), matA_rowptr);
        PonterUtils.copyToPointer(matrixA.getColdata(), matA_colindices);
        PonterUtils.copyToPointer(matrixB.getRowptr(), matB_rowptr);
        PonterUtils.copyToPointer(matrixB.getColdata(), matB_colindices);
        if (!isBinary) {
        	PonterUtils.copyToPointer(matrixA.getData(), matA_data);
        	PonterUtils.copyToPointer(matrixB.getData(), matB_data);
        }
        
        
        // CLBuffers
        CLBuffer<Integer> cl_counter = null, cl_matA_rowptr = null, cl_matA_colindices = null, cl_matB_rowptr = null, cl_matB_colindices = null;
        CLBuffer<Float> cl_matA_data = null, cl_matB_data = null;
        CLBuffer<Integer> cl_output_data_x = null, cl_output_data_y = null;
        CLBuffer<Float> cl_output_data_val = null;
        
        try {
        	// Always use device mem for the counter
        	cl_counter = context.createBuffer(Usage.InputOutput, counter);
        	buffersRelease.add(cl_counter);
            cl_matA_rowptr = context.createBuffer(Usage.Input, matA_rowptr, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matA_rowptr);
            cl_matA_colindices = context.createBuffer(Usage.Input, matA_colindices, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matA_colindices);
            cl_matB_rowptr = context.createBuffer(Usage.Input, matB_rowptr, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matB_rowptr);
            cl_matB_colindices = context.createBuffer(Usage.Input, matB_colindices, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matB_colindices);
            if (!isBinary) {
            	cl_matA_data = context.createBuffer(Usage.Input, matA_data, CLEngineConfig.isUSE_DEVICE_MEM());
            	buffersRelease.add(cl_matA_data);
            	cl_matB_data = context.createBuffer(Usage.Input, matB_data, CLEngineConfig.isUSE_DEVICE_MEM());
            	buffersRelease.add(cl_matB_data);
            }
            
            // Output buffer
            cl_output_data_x = context.createIntBuffer(Usage.Output, nnzCount);
            buffersRelease.add(cl_output_data_x);
            cl_output_data_y = context.createIntBuffer(Usage.Output, nnzCount);
            buffersRelease.add(cl_output_data_y);
            cl_output_data_val = context.createFloatBuffer(Usage.Output, nnzCount);
            buffersRelease.add(cl_output_data_val);            
        } catch (CLException e) {
        	queue.flush();
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);
			context.release();
			
			System.err.println(e.toString());
			return null;    	
        }


        // Read the program sources and compile them :
        String kernelSource = null;
		try {
			kernelSource = IOUtils.readText(MultiplyCL.class.getResource( KernelConfig.KERNEL_COO_FLAT() ));
		} catch (IOException e) {
			queue.flush();
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);
			context.release();
			
			System.err.println(e.toString());
			return null;
		}
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_ROW, Integer.toString( matrixA.getRowCount() ) );
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_COL, Integer.toString( matrixB.getRowCount() ) );
        
    	// System.out.println(kernelSource);
        
        CLProgram program = context.createProgram(kernelSource);

        // Get and call the kernel :
        CLKernel multiplyMatrixKernel = null;
        if (!isBinary) {
        	multiplyMatrixKernel = program.createKernel(KernelConfig.KERNEL_COO_FUN_FULL);
        	multiplyMatrixKernel.setArgs(cl_matA_rowptr,
        			cl_matA_colindices,
        			cl_matA_data,
        			cl_matB_rowptr,
        			cl_matB_colindices,
        			cl_matB_data,
        			cl_counter,
        			cl_output_data_x,
        			cl_output_data_y,
        			cl_output_data_val);
        } else {
        	multiplyMatrixKernel = program.createKernel(KernelConfig.KERNEL_COO_FUN_SHORT);
        	multiplyMatrixKernel.setArgs(cl_matA_rowptr,
        			cl_matA_colindices,
        			cl_matB_rowptr,
        			cl_matB_colindices,
        			cl_counter,
        			cl_output_data_x,
        			cl_output_data_y,
        			cl_output_data_val);
        }
        
        int[] wgSize;
        int[] locSize;
        
        if (CLEngineConfig.isIMPL_LOCAL()) {
        	wgSize = new int[]{matrixA.getRowCount()};
        	locSize = null;
        } else {
    		try {
    			List<int[]> niceSizes = SizeEstimator.getGoodSizes(matrixA.getRowCount(), 1, (int) maxWorkGroupSize);
    			wgSize = new int[]{niceSizes.get(0)[0]};
    			locSize = new int[]{niceSizes.get(1)[0]};
    		} catch (Exception e) {
    			queue.flush();
    			queue.release();
    			multiplyMatrixKernel.release();
    			program.release();
    			clearAllocatedCLObjects(buffersRelease);
    			clearAllocatedPTRObjects(pointersRelease);
    			context.release();
    			
    			System.err.println(e.toString());
    			return null;
    		}        	
        }

//        System.err.println(wgSize[0]);
        CLEvent addEvt = null;
        if (CLEngineConfig.isIMPL_LOCAL()) {
        	addEvt = multiplyMatrixKernel.enqueueNDRange(queue, wgSize);
        } else {
        	addEvt = multiplyMatrixKernel.enqueueNDRange(queue, wgSize, locSize);
        }
       
        Pointer<Integer> matrixDataOut_x = cl_output_data_x.read(queue, addEvt);
        pointersRelease.add(matrixDataOut_x);
        Pointer<Integer> matrixDataOut_y = cl_output_data_y.read(queue, addEvt);
        pointersRelease.add(matrixDataOut_y);
        Pointer<Float> matrixDataOut_val = cl_output_data_val.read(queue, addEvt);
        pointersRelease.add(matrixDataOut_val);
        // Pointer<Float> matrixDataOut = Pointer.allocateFloats(matrixA.getRowCount()*matrixBToTranspose.getColCount()).order(byteOrder);
        // cl_output_data.read(queue, matrixDataOut, true, addEvt);
        
        List<Integer> listMatrixOut_x = PonterUtils.copyFromPointerInteger(matrixDataOut_x);
        List<Integer> listMatrixOut_y = PonterUtils.copyFromPointerInteger(matrixDataOut_y);
        List<Float> listMatrixOut_val = PonterUtils.copyFromPointerFloat(matrixDataOut_val);
		
        addEvt.release();
        queue.flush();
        queue.release();
		multiplyMatrixKernel.release();
		program.release();
		clearAllocatedCLObjects(buffersRelease);
		clearAllocatedPTRObjects(pointersRelease);
		context.release();
		
		System.err.println("Calculated in: " + (System.currentTimeMillis() - startTime) + " millis");
		
//		System.out.println(listMatrixOut_x);
//		System.out.println(listMatrixOut_y);
//		System.out.println(listMatrixOut_val);
		
		return CsrMatrix.fromCOOArray(listMatrixOut_x, listMatrixOut_y, listMatrixOut_val, matrixA.getRowshape(), matrixBToTranspose.getColshape());
	}
	
	
	@SuppressWarnings("unused")
	private static int clCalcNNZ(CsrMatrix matrixA, CsrMatrix matrixBToTranspose) {
		// Lista di CL buffer da deallocare
		List<CLMem> buffersRelease = Lists.newArrayList();
		@SuppressWarnings("rawtypes")
		List<Pointer> pointersRelease = Lists.newArrayList();
		
		long startTime = System.currentTimeMillis();
		//
		CLContext context = KernelConfig.createContext();
		
		if (context == null) {
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);

			return -1;    	
        }
		// Context

		
		// WorkGroupSize
		long maxWorkGroupSize = Long.MAX_VALUE;
		for(CLDevice currDev: context.getDevices() ) {
			maxWorkGroupSize = Math.min(maxWorkGroupSize, currDev.getMaxWorkGroupSize());
		}
		
        CLQueue queue = context.createDefaultQueue();
        ByteOrder byteOrder = context.getByteOrder();
        
        CsrMatrix matrixB = matrixBToTranspose.transpose();
        
        // Native memory
        Pointer<Integer> counter, matA_rowptr, matA_colindices, matB_rowptr, matB_colindices;
        
        // Allocate
        counter = Pointer.allocateInt().order(byteOrder);
        counter.set(0);
        pointersRelease.add(counter);
        matA_rowptr = Pointer.allocateInts(matrixA.getRowptr().size()).order(byteOrder);
        pointersRelease.add(matA_rowptr);
        matA_colindices = Pointer.allocateInts(matrixA.getColdata().size()).order(byteOrder);
        pointersRelease.add(matA_colindices);
        matB_rowptr = Pointer.allocateInts(matrixB.getRowptr().size()).order(byteOrder);
        pointersRelease.add(matB_rowptr);
        matB_colindices = Pointer.allocateInts(matrixB.getColdata().size()).order(byteOrder);
        pointersRelease.add(matB_colindices);
        
        PonterUtils.copyToPointer(matrixA.getRowptr(), matA_rowptr);
        PonterUtils.copyToPointer(matrixA.getColdata(), matA_colindices);
        PonterUtils.copyToPointer(matrixB.getRowptr(), matB_rowptr);
        PonterUtils.copyToPointer(matrixB.getColdata(), matB_colindices);
        
        
        // CLBuffers
        CLBuffer<Integer> cl_counter = null, cl_matA_rowptr = null, cl_matA_colindices = null, cl_matB_rowptr = null, cl_matB_colindices = null;
        
        try {
        	// Always use device mem for the counter
        	cl_counter = context.createBuffer(Usage.InputOutput, counter);
        	buffersRelease.add(cl_counter);
            cl_matA_rowptr = context.createBuffer(Usage.Input, matA_rowptr, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matA_rowptr);
            cl_matA_colindices = context.createBuffer(Usage.Input, matA_colindices, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matA_colindices);
            cl_matB_rowptr = context.createBuffer(Usage.Input, matB_rowptr, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matB_rowptr);
            cl_matB_colindices = context.createBuffer(Usage.Input, matB_colindices, CLEngineConfig.isUSE_DEVICE_MEM());
            buffersRelease.add(cl_matB_colindices);
        } catch (CLException e) {
        	queue.flush();
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);
			context.release();
			
			System.err.println(e.toString());
			return -1;    	
        }


        // Read the program sources and compile them :
        String kernelSource = null;
		try {
			kernelSource = IOUtils.readText( MultiplyCL.class.getResource(KernelConfig.KERNEL_NNZ() ));
		} catch (IOException e) {
			queue.flush();
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			clearAllocatedPTRObjects(pointersRelease);
			context.release();
			
			System.err.println(e.toString());
			return -1;
		}
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_ROW, Integer.toString( matrixA.getRowCount() ) );
    	kernelSource = kernelSource.replaceAll(KernelConfig.DEFINE_COL, Integer.toString( matrixB.getRowCount() ) );
        
    	// System.out.println(kernelSource);
        
        CLProgram program = context.createProgram(kernelSource);

        // Get and call the kernel :
        CLKernel multiplyMatrixKernel = null;

       	multiplyMatrixKernel = program.createKernel(KernelConfig.KERNEL_NNZ_FUN);
       	multiplyMatrixKernel.setArgs(cl_matA_rowptr,
        			cl_matA_colindices,
        			cl_matB_rowptr,
        			cl_matB_colindices,
        			cl_counter);
        
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
    			clearAllocatedCLObjects(buffersRelease);
    			clearAllocatedPTRObjects(pointersRelease);
    			context.release();
    			
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
       
        counter = cl_counter.read(queue, addEvt);
        // Pointer<Float> matrixDataOut = Pointer.allocateFloats(matrixA.getRowCount()*matrixBToTranspose.getColCount()).order(byteOrder);
        // cl_output_data.read(queue, matrixDataOut, true, addEvt);
        
        int resultCount = counter.get();
		
        addEvt.release();
        queue.flush();
        queue.release();
		multiplyMatrixKernel.release();
		program.release();
		clearAllocatedCLObjects(buffersRelease);
		clearAllocatedPTRObjects(pointersRelease);
		context.release();
		
		System.err.println("Calculated in: " + (System.currentTimeMillis() - startTime) + " millis");
        
        return resultCount;
	}
	
	private static void clearAllocatedCLObjects(List<CLMem> listOfObjects) {
		System.err.println("Clearing CLMEM");
		for(CLMem buffObject: listOfObjects) {
			buffObject.release();
		}
		listOfObjects.clear();
	}
	
	@SuppressWarnings("rawtypes")
	private static void clearAllocatedPTRObjects(List<Pointer> listOfObjects) {
		System.err.println("Clearing POINTERS");
		for(Pointer buffObject: listOfObjects) {
			buffObject.release();
		}
		listOfObjects.clear();
	}	
	
	public static void main(String[] args) throws Exception {
		int[] matrixOne = new int[]{1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0};
		int[] matrixTwo = new int[]{1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,1,1,0,1};
		CsrMatrix csrMatrixOne = CsrMatrix.fromFlattenArray(matrixOne, 5);
		CsrMatrix csrMatrixTwo = CsrMatrix.fromFlattenArray(matrixTwo, 4);
		System.out.println(csrMatrixOne);
		System.out.println(csrMatrixTwo);
		System.out.println(csrMatrixTwo.transpose());
		System.out.println("==========");
		System.out.println(csrMatrixOne.multiply(csrMatrixTwo));	
		
//		
//		CsrMatrix result = multiply(csrMatrixOne, csrMatrixTwo, false);
//		System.out.println(result);
//		System.out.println(csrMatrixOne.multiply(csrMatrixTwo));
////		System.out.println(csrMatrixOne.transpose());
//		System.out.println("==========");
//		
//		System.out.println(KernelConfig.KERNEL_COO_FLAT());
//		System.out.println(clMultiplyCOO_FLAT(csrMatrixOne, csrMatrixTwo, clCalcNNZ(csrMatrixOne, csrMatrixTwo)));
//		System.out.println(csrMatrixOne.multiply(csrMatrixTwo));
//		
////		float[] ccoOutput = new float[]{0, 0, 2, 
////										2, 0, 1, 
////										0, 1, 1, 
////										1, 1, 1, 
////										2, 1, 1, 
////										0, 3, 2, 
////										2, 3, 1, 
////										3, 3, 2};
//		
////		System.out.println(
////				CsrMatrix.fromCOOArray(ccoOutput, csrMatrixOne.getRowshape(), csrMatrixTwo.getColshape())
////				);
		
//		System.out.println(csrMatrixOne.nnzMultiplyCount(csrMatrixTwo));
//		System.out.println(clCalcNNZ(csrMatrixOne, csrMatrixTwo));
	}
}

/*
JavaCL maps OpenCL entities (allocated by the OpenCL driver, typically in the device memory) 
to Java objects (managed by the JVM's garbage collector).

OpenCL entities are released when their Java object counterparts are garbage collected or when 
their release() method is called.

In many cases, waiting for the GC to do the work can lead to serious issues : when the OpenCL 
driver runs out of memory, it does not tell Java to try and collect unused objects 
(which would release a few OpenCL entities in the process) and just fails, which makes JavaCL 
throw a CLException.MemObjectAllocationFailure or CLException.OutOfResources exception.

To avoid that, one can manually release an unused buffer (or any JavaCL entity) by calling 
CLAbstractEntity.release() (CLAbstractEntity is a base class which is inherited by CLBuffer, 
CLImage2D, CLProgram, CLEvent... virtually all JavaCL classes of interest).

Fortunately, JavaCL features a workaround for allocations : whenever they fail by lack of 
OpenCL memory, JavaCL triggers a full GC, waits a little while and retries. 
This might have a terribly negative impact on your application's performance, though, so please 
call release() as soon as you can!
*/ 
