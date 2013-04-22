package it.cvdlab.lar.clengine;

import it.cvdlab.lar.model.CsrMatrix;

import java.io.IOException;
import java.nio.ByteOrder;
import java.util.List;

import org.bridj.Pointer;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLPlatform.DeviceFeature;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.util.IOUtils;

public class MultiplyCL {
	public static synchronized CsrMatrix multiply(CsrMatrix matrixA, CsrMatrix matrixB) {
		return clMultiply(matrixA, matrixB);
	}
	
	private static CsrMatrix clMultiply (CsrMatrix matrixA, CsrMatrix matrixBx) {
		CLContext context = JavaCL.createBestContext(DeviceFeature.GPU);
		long maxWorkGroupSize = Long.MAX_VALUE;
		for(CLDevice currDev: context.getDevices() ) {
			maxWorkGroupSize = Math.min(maxWorkGroupSize, currDev.getMaxWorkGroupSize());
		}
				
        CLQueue queue = context.createDefaultQueue();
        ByteOrder byteOrder = context.getByteOrder();
        
        CsrMatrix matrixB = matrixBx.transpose();
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
        CLBuffer<Integer> cl_matA_rowptr, cl_matA_colindices, cl_matB_rowptr, cl_matB_colindices;
        CLBuffer<Float> cl_matA_data = null, cl_matB_data = null;
        
        cl_matA_rowptr = context.createBuffer(Usage.Input, matA_rowptr);
        cl_matA_colindices = context.createBuffer(Usage.Input, matA_colindices);
        cl_matB_rowptr = context.createBuffer(Usage.Input, matB_rowptr);
        cl_matB_colindices = context.createBuffer(Usage.Input, matB_colindices);
        if (!isBinary) {
        	cl_matA_data = context.createBuffer(Usage.Input, matA_data);
        	cl_matB_data = context.createBuffer(Usage.Input, matB_data);
        }
        
        // Output buffer
        CLBuffer<Float> cl_output_data = context.createFloatBuffer(Usage.Output, matrixA.getRowCount()*matrixBx.getColCount());

        // Read the program sources and compile them :
        String kernelSource = null;
		try {
			kernelSource = IOUtils.readText(MultiplyCL.class.getResource("SpMSpM-Multiply-Naive.cl"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
    	kernelSource = kernelSource.replaceAll("%%AROW%%", Integer.toString( matrixA.getRowCount() ) );
    	kernelSource = kernelSource.replaceAll("%%BCOL%%", Integer.toString( matrixB.getRowCount() ) );
        
        
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
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
        
        CLEvent addEvt = multiplyMatrixKernel.enqueueNDRange(queue, niceSizes.get(0), niceSizes.get(1));
        
        Pointer<Float> matrixDataOut = cl_output_data.read(queue, addEvt);

		return null;
	}
	
	
	private static <T> void copyToPointer(List<T> iList, Pointer<T> oPointer) {
		for(int i = 0; i < iList.size(); i++) {
			oPointer.set(i, iList.get(i));
		}
	}
}

/*
JavaCL maps OpenCL entities (allocated by the OpenCL driver, typically in the device memory) to Java objects (managed by the JVM's garbage collector).

OpenCL entities are released when their Java object counterparts are garbage collected or when their release() method is called.

In many cases, waiting for the GC to do the work can lead to serious issues : when the OpenCL driver runs out of memory, it does not tell Java to try and collect unused objects (which would release a few OpenCL entities in the process) and just fails, which makes JavaCL throw a CLException.MemObjectAllocationFailure or CLException.OutOfResources exception.

To avoid that, one can manually release an unused buffer (or any JavaCL entity) by calling CLAbstractEntity.release() (CLAbstractEntity is a base class which is inherited by CLBuffer, CLImage2D, CLProgram, CLEvent... virtually all JavaCL classes of interest).

Fortunately, JavaCL features a workaround for allocations : whenever they fail by lack of OpenCL memory, JavaCL triggers a full GC, waits a little while and retries. This might have a terribly negative impact on your application's performance, though, so please call release() as soon as you can!
*/ 
