package it.cvdlab.lar.clengine;

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
import com.nativelibs4java.opencl.CLPlatform.DeviceFeature;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.util.IOUtils;

public class MultiplyCL {
	private static final Logger logger = LoggerFactory.getLogger(MultiplyCL.class);
	private static final int NNZ_WEIGHT = 1;
	
	public static synchronized CsrMatrix multiply(CsrMatrix matrixA, CsrMatrix matrixB) {
		// First calculate NNZ elements!! This is necessary!
		int nnzCount = -1;
		
		try {
			nnzCount = matrixA.nnzMultiplyCount(matrixB);
			logger.info("NNZCount: " + nnzCount );
		} catch (Exception e1) {
			logger.error(e1.toString());
			return null; 
		}		
		
		if ((matrixA.getRowCount() * matrixB.getColCount()) > ( nnzCount * NNZ_WEIGHT )) {
			logger.info("COO Way");
			return clMultiplyCOO(matrixA, matrixB, nnzCount);
		} else {
			System.out.println("Dense Way");
			return clMultiply(matrixA, matrixB);
		}
	}
	
	private static CsrMatrix clMultiply(CsrMatrix matrixA, CsrMatrix matrixBToTranspose) {
		// Lista di CL buffer da deallocare
		List<CLMem> buffersRelease = Lists.newArrayList();

		// Context
		CLContext context = JavaCL.createBestContext(DeviceFeature.GPU);
		// CLContext context = JavaCL.createBestContext();
		
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
        matA_colindices = Pointer.allocateInts(matrixA.getColdata().size()).order(byteOrder);
        matB_rowptr = Pointer.allocateInts(matrixB.getRowptr().size()).order(byteOrder);
        matB_colindices = Pointer.allocateInts(matrixB.getColdata().size()).order(byteOrder);
        if (!isBinary) {
        	matA_data = Pointer.allocateFloats(matrixA.getData().size()).order(byteOrder);
        	matB_data = Pointer.allocateFloats(matrixB.getData().size()).order(byteOrder);
        }
        
        copyToPointer(matrixA.getRowptr(), matA_rowptr);
        copyToPointer(matrixA.getColdata(), matA_colindices);
        copyToPointer(matrixB.getRowptr(), matB_rowptr);
        copyToPointer(matrixB.getColdata(), matB_colindices);
        if (!isBinary) {
        	copyToPointer(matrixA.getData(), matA_data);
        	copyToPointer(matrixB.getData(), matB_data);
        }
        
        
        // CLBuffers
        CLBuffer<Integer> cl_matA_rowptr = null, cl_matA_colindices = null, cl_matB_rowptr = null, cl_matB_colindices = null;
        CLBuffer<Float> cl_matA_data = null, cl_matB_data = null;
        CLBuffer<Float> cl_output_data = null;
        
        try {
            cl_matA_rowptr = context.createBuffer(Usage.Input, matA_rowptr);
            buffersRelease.add(cl_matA_rowptr);
            cl_matA_colindices = context.createBuffer(Usage.Input, matA_colindices);
            buffersRelease.add(cl_matA_colindices);
            cl_matB_rowptr = context.createBuffer(Usage.Input, matB_rowptr);
            buffersRelease.add(cl_matB_rowptr);
            cl_matB_colindices = context.createBuffer(Usage.Input, matB_colindices);
            buffersRelease.add(cl_matB_colindices);
            if (!isBinary) {
            	cl_matA_data = context.createBuffer(Usage.Input, matA_data);
            	buffersRelease.add(cl_matA_data);
            	cl_matB_data = context.createBuffer(Usage.Input, matB_data);
            	buffersRelease.add(cl_matB_data);
            }
            
            // Output buffer
            cl_output_data = context.createFloatBuffer(Usage.Output, matrixA.getRowCount()*matrixBToTranspose.getColCount());
            buffersRelease.add(cl_output_data);
        } catch (CLException e) {
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			
			logger.error(e.toString());
			return null;        	
        }


        // Read the program sources and compile them :
        String kernelSource = null;
		try {
			kernelSource = IOUtils.readText(MultiplyCL.class.getResource("SpMSpM-Multiply-Naive.cl"));
		} catch (IOException e) {
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			
			logger.error(e.toString());
			return null;
		}
    	kernelSource = kernelSource.replaceAll("%%AROW%%", Integer.toString( matrixA.getRowCount() ) );
    	kernelSource = kernelSource.replaceAll("%%BCOL%%", Integer.toString( matrixB.getRowCount() ) );
        
    	// System.out.println(kernelSource);
        
        CLProgram program = context.createProgram(kernelSource);

        // Get and call the kernel :
        CLKernel multiplyMatrixKernel = null;
        if (!isBinary) {
        	multiplyMatrixKernel = program.createKernel("spmm_kernel_naive");
        	multiplyMatrixKernel.setArgs(cl_matA_rowptr,
        			cl_matA_colindices,
        			cl_matA_data,
        			cl_matB_rowptr,
        			cl_matB_colindices,
        			cl_matB_data,
        			cl_output_data);
        } else {
        	multiplyMatrixKernel = program.createKernel("spmm_binary_kernel_naive");
        	multiplyMatrixKernel.setArgs(cl_matA_rowptr,
        			cl_matA_colindices,
        			cl_matB_rowptr,
        			cl_matB_colindices,
        			cl_output_data);
        }
        
        List<int[]> niceSizes;
		try {
			niceSizes = SizeEstimator.getGoodSizes(matrixA.getRowCount(), matrixB.getRowCount(), (int) maxWorkGroupSize);
		} catch (Exception e) {
			queue.release();
			multiplyMatrixKernel.release();
			program.release();
			clearAllocatedCLObjects(buffersRelease);
			
			logger.error(e.toString());
			return null;
		}

        // queue.finish();
        CLEvent addEvt = multiplyMatrixKernel.enqueueNDRange(queue, niceSizes.get(0), niceSizes.get(1));
        
       
        Pointer<Float> matrixDataOut = cl_output_data.read(queue, addEvt);
        // Pointer<Float> matrixDataOut = Pointer.allocateFloats(matrixA.getRowCount()*matrixBToTranspose.getColCount()).order(byteOrder);
        // cl_output_data.read(queue, matrixDataOut, true, addEvt);
        
        List<Float> listMatrixOut = copyFromPointer(matrixDataOut);
        
		queue.release();
		multiplyMatrixKernel.release();
		program.release();
		clearAllocatedCLObjects(buffersRelease);
        
		// System.out.println(listMatrixOut);
		
		return CsrMatrix.fromFlattenArray(ArrayUtils.toPrimitive( listMatrixOut.toArray(new Float[0]) ), matrixBToTranspose.getColCount());
	}
	
	public static CsrMatrix clMultiplyCOO(CsrMatrix matrixA, CsrMatrix matrixBToTranspose, int nnzCount) {
		// Lista di CL buffer da deallocare
		List<CLMem> buffersRelease = Lists.newArrayList();

		// Context
		CLContext context = JavaCL.createBestContext(DeviceFeature.GPU);
//		CLContext context = JavaCL.createBestContext();
		
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
        matA_rowptr = Pointer.allocateInts(matrixA.getRowptr().size()).order(byteOrder);
        matA_colindices = Pointer.allocateInts(matrixA.getColdata().size()).order(byteOrder);
        matB_rowptr = Pointer.allocateInts(matrixB.getRowptr().size()).order(byteOrder);
        matB_colindices = Pointer.allocateInts(matrixB.getColdata().size()).order(byteOrder);
        if (!isBinary) {
        	matA_data = Pointer.allocateFloats(matrixA.getData().size()).order(byteOrder);
        	matB_data = Pointer.allocateFloats(matrixB.getData().size()).order(byteOrder);
        }
        
        copyToPointer(matrixA.getRowptr(), matA_rowptr);
        copyToPointer(matrixA.getColdata(), matA_colindices);
        copyToPointer(matrixB.getRowptr(), matB_rowptr);
        copyToPointer(matrixB.getColdata(), matB_colindices);
        if (!isBinary) {
        	copyToPointer(matrixA.getData(), matA_data);
        	copyToPointer(matrixB.getData(), matB_data);
        }
        
        
        // CLBuffers
        CLBuffer<Integer> cl_counter = null, cl_matA_rowptr = null, cl_matA_colindices = null, cl_matB_rowptr = null, cl_matB_colindices = null;
        CLBuffer<Float> cl_matA_data = null, cl_matB_data = null;
        CLBuffer<Float> cl_output_data = null;
        
        try {
        	cl_counter = context.createBuffer(Usage.InputOutput, counter);
        	buffersRelease.add(cl_counter);
            cl_matA_rowptr = context.createBuffer(Usage.Input, matA_rowptr);
            buffersRelease.add(cl_matA_rowptr);
            cl_matA_colindices = context.createBuffer(Usage.Input, matA_colindices);
            buffersRelease.add(cl_matA_colindices);
            cl_matB_rowptr = context.createBuffer(Usage.Input, matB_rowptr);
            buffersRelease.add(cl_matB_rowptr);
            cl_matB_colindices = context.createBuffer(Usage.Input, matB_colindices);
            buffersRelease.add(cl_matB_colindices);
            if (!isBinary) {
            	cl_matA_data = context.createBuffer(Usage.Input, matA_data);
            	buffersRelease.add(cl_matA_data);
            	cl_matB_data = context.createBuffer(Usage.Input, matB_data);
            	buffersRelease.add(cl_matB_data);
            }
            
            // Output buffer
            cl_output_data = context.createFloatBuffer(Usage.Output, nnzCount*3); //float3
            buffersRelease.add(cl_output_data);
        } catch (CLException e) {
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			
			logger.error(e.toString());
			return null;    	
        }


        // Read the program sources and compile them :
        String kernelSource = null;
		try {
			kernelSource = IOUtils.readText(MultiplyCL.class.getResource("SpMSpM-Multiply-COO.cl"));
		} catch (IOException e) {
			queue.release();
			clearAllocatedCLObjects(buffersRelease);
			
			logger.error(e.toString());
			return null;
		}
    	kernelSource = kernelSource.replaceAll("%%AROW%%", Integer.toString( matrixA.getRowCount() ) );
    	kernelSource = kernelSource.replaceAll("%%BCOL%%", Integer.toString( matrixB.getRowCount() ) );
        
    	// System.out.println(kernelSource);
        
        CLProgram program = context.createProgram(kernelSource);

        // Get and call the kernel :
        CLKernel multiplyMatrixKernel = null;
        if (!isBinary) {
        	multiplyMatrixKernel = program.createKernel("spmm_coo_kernel_naive");
        	multiplyMatrixKernel.setArgs(cl_matA_rowptr,
        			cl_matA_colindices,
        			cl_matA_data,
        			cl_matB_rowptr,
        			cl_matB_colindices,
        			cl_matB_data,
        			cl_counter,
        			cl_output_data);
        } else {
        	multiplyMatrixKernel = program.createKernel("spmm_coo_binary_kernel_naive");
        	multiplyMatrixKernel.setArgs(cl_matA_rowptr,
        			cl_matA_colindices,
        			cl_matB_rowptr,
        			cl_matB_colindices,
        			cl_counter,
        			cl_output_data);
        }
        
        List<int[]> niceSizes;
		try {
			niceSizes = SizeEstimator.getGoodSizes(matrixA.getRowCount(), matrixB.getRowCount(), (int) maxWorkGroupSize);
		} catch (Exception e) {
			queue.release();
			multiplyMatrixKernel.release();
			program.release();
			clearAllocatedCLObjects(buffersRelease);
			
			logger.error(e.toString());
			return null;
		}

        // queue.finish();
        CLEvent addEvt = multiplyMatrixKernel.enqueueNDRange(queue, niceSizes.get(0), niceSizes.get(1));
        
       
        Pointer<Float> matrixDataOut = cl_output_data.read(queue, addEvt);
        // Pointer<Float> matrixDataOut = Pointer.allocateFloats(matrixA.getRowCount()*matrixBToTranspose.getColCount()).order(byteOrder);
        // cl_output_data.read(queue, matrixDataOut, true, addEvt);
        
        List<Float> listMatrixOut = copyFromPointer(matrixDataOut);
        
		queue.release();
		multiplyMatrixKernel.release();
		program.release();
		clearAllocatedCLObjects(buffersRelease);
		
		return CsrMatrix.fromCOOArray(listMatrixOut, matrixA.getRowshape(), matrixBToTranspose.getColshape());
	}	
	
	
	private static <T> void copyToPointer(List<T> iList, Pointer<T> oPointer) {
		for(int i = 0; i < iList.size(); i++) {
			oPointer.set(i, iList.get(i));
		}
	}
	
	private static <T> List<T> copyFromPointer(Pointer<T> iPointer) {
		List<T> tmpList = Lists.newArrayList();
		for(T singleData: iPointer) {
			tmpList.add(singleData);
		}
		return Lists.newArrayList(tmpList);
	}	
	
	private static void clearAllocatedCLObjects(List<CLMem> listOfObjects) {
		for(CLMem buffObject: listOfObjects) {
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
		System.out.println("==========");
		
		CsrMatrix result = multiply(csrMatrixOne, csrMatrixTwo);
		System.out.println(result);
		System.out.println(csrMatrixOne.multiply(csrMatrixTwo));
//		System.out.println(csrMatrixOne.transpose());
		System.out.println("==========");
		
//		clMultiplyCOO(csrMatrixOne, csrMatrixTwo);
//		System.out.println(csrMatrixOne.multiply(csrMatrixTwo));
		
//		float[] ccoOutput = new float[]{0, 0, 2, 
//										2, 0, 1, 
//										0, 1, 1, 
//										1, 1, 1, 
//										2, 1, 1, 
//										0, 3, 2, 
//										2, 3, 1, 
//										3, 3, 2};
		
//		System.out.println(
//				CsrMatrix.fromCOOArray(ccoOutput, csrMatrixOne.getRowshape(), csrMatrixTwo.getColshape())
//				);
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
