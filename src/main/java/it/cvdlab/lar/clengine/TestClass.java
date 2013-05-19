package it.cvdlab.lar.clengine;

import it.cvdlab.lar.model.CsrMatrix;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.codehaus.jackson.map.ObjectMapper;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.CLPlatform.DeviceFeature;
import com.nativelibs4java.util.IOUtils;

final class TestClass {
	// CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		String matrixB = FileUtils.readFileToString(new File("matrixb.1368997187580.log"));
		CsrMatrix secondMatrix = jacksonMapper.readValue(matrixB , CsrMatrix.class);
		System.out.println("NNZ: " + secondMatrix.getNonZeroElementsCount());
		CsrMatrix sT = secondMatrix.transpose();
		System.out.println("NNZ: " + sT.getNonZeroElementsCount());
		writeLogMatrix("matrixbtransp", jacksonMapper.writeValueAsString(sT));
		System.out.println( sT.transpose().equals(secondMatrix) );
		
	}
	
	private static ObjectMapper jacksonMapper = new ObjectMapper();
	
    private static void writeLogMatrix(String name, String content){
        try {
            FileWriter fw = new FileWriter(name + "." + System.currentTimeMillis() + ".log");
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(content);
            bw.close();
        } catch (IOException e) {
            System.err.print("Unable to write to file " + name+ ".");
            e.printStackTrace();
        }
    }  	
}
