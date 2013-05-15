package it.cvdlab.lar.rest;

import it.cvdlab.lar.clengine.MultiplyCL;
import it.cvdlab.lar.model.CsrMatrix;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URLDecoder;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.ws.rs.Consumes;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.MultivaluedMap;
import javax.ws.rs.core.UriInfo;

import org.codehaus.jackson.JsonParseException;
import org.codehaus.jackson.map.JsonMappingException;
import org.codehaus.jackson.map.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Path(RestService.REST_SERVICE_URL)
public class RestService {
	private static final String MATRIX_FIRST_PARAM = "matrixa";
	private static final String MATRIX_SECOND_PARAM = "matrixb";
	
	
    @Context
    private HttpServletRequest httpServletRequest;
    @Context
    private HttpServletResponse httpServletResponse;
    @Context
    private UriInfo uriInfo;
    // Jackson
    private ObjectMapper jacksonMapper = new ObjectMapper();
    
	private static final Logger logger = LoggerFactory.getLogger(RestService.class);
    public static final String REST_SERVICE_URL = "/multiply";
    
    // httpServletResponse.addHeader("Access-Control-Allow-Origin", "*");

    // /lar/services/multiply/execute
    @Path("/execute")
    @POST
    @Consumes({ MediaType.APPLICATION_FORM_URLENCODED })
    @Produces({ MediaType.APPLICATION_JSON })
    public CsrMatrix doMultiply(@Context UriInfo uriInfo, MultivaluedMap<String, String> form) {
    	// httpServletResponse.addHeader("Access-Control-Allow-Origin", "*");
    	logger.error("/execute");
    	return computeProduct(form, false);
    }
    
    @Path("/executeCOO")
    @POST
    @Consumes({ MediaType.APPLICATION_FORM_URLENCODED })
    @Produces({ MediaType.APPLICATION_JSON })
    public CsrMatrix doMultiplyCOO(@Context UriInfo uriInfo, MultivaluedMap<String, String> form) {
    	// httpServletResponse.addHeader("Access-Control-Allow-Origin", "*");
    	logger.error("/executeCOO");
    	return computeProduct(form, true);
    }    
    
    @Path("/serialize")
    @POST
    @Consumes({ MediaType.APPLICATION_FORM_URLENCODED })
    @Produces({ MediaType.APPLICATION_JSON })
    public CsrMatrix doSerialize(@Context UriInfo uriInfo, MultivaluedMap<String, String> form) {
    	CsrMatrix firstMatrix = null;
    	
    	System.err.println(form.toString());
    	
    	if ( form.containsKey(MATRIX_FIRST_PARAM) ) {
    		try {
    			firstMatrix = jacksonMapper.readValue(form.getFirst(MATRIX_FIRST_PARAM), CsrMatrix.class);
			} catch (JsonParseException e) {
				System.err.println( e.toString() );
			} catch (JsonMappingException e) {
				System.err.println( e.toString() );
			} catch (IOException e) {
				System.err.println( e.toString() );
			}
    	}
    	
    	System.err.println(firstMatrix.isBinary());
    	
    	return firstMatrix;
    }
    
    @Path("/networktest")
    @GET
    @Produces({ MediaType.APPLICATION_JSON })
    public String doTest() {
    	// httpServletResponse.addHeader("Access-Control-Allow-Origin", "*");

        return (new CsrMatrix(new int[]{0,1,2}, new int[]{0,1}, 2, 2)).toDense().toString();    	
    }
    
    
    private CsrMatrix computeProduct(MultivaluedMap<String, String> form, boolean forceCOO) {
    	String matrixContent = null;
    	CsrMatrix firstMatrix = null;
    	CsrMatrix secondMatrix = null;
    	boolean firstParse = false;
    	boolean secondParse = false;

    	// System.err.println(form.toString());
    	
    	if ( form.containsKey(MATRIX_FIRST_PARAM) ) {
    		try {
    			matrixContent = URLDecoder.decode(form.getFirst(MATRIX_FIRST_PARAM), "UTF-8");
    			// writeLogMatrix(MATRIX_FIRST_PARAM, matrixContent);
    			firstMatrix = jacksonMapper.readValue(matrixContent, CsrMatrix.class);
    			firstParse = true;
			} catch (JsonParseException e) {
				System.err.println( e.toString() );
			} catch (JsonMappingException e) {
				System.err.println( e.toString() );
			} catch (IOException e) {
				System.err.println( e.toString() );
			}
    	}
    	
    	matrixContent = null;
    	
    	if ( form.containsKey(MATRIX_SECOND_PARAM) ) {
    		try {
    			matrixContent = URLDecoder.decode(form.getFirst(MATRIX_SECOND_PARAM), "UTF-8");
    			// writeLogMatrix(MATRIX_SECOND_PARAM, matrixContent);    			
    			secondMatrix = jacksonMapper.readValue(matrixContent , CsrMatrix.class);
    			secondParse = true;
			} catch (JsonParseException e) {
				System.err.println( e.toString() );
			} catch (JsonMappingException e) {
				System.err.println( e.toString() );
			} catch (IOException e) {
				System.err.println( e.toString() );
			}
    	}
    	
    	CsrMatrix resultMatrix = null;
    	if ((firstMatrix != null) && (secondMatrix != null) && firstParse && secondParse) {
    		System.err.println("Starting RESULT matrix..."); 
    		resultMatrix = MultiplyCL.multiply(firstMatrix, secondMatrix, forceCOO);
    		System.err.println("Sending RESULT matrix..."); 
    	}
    	
        return resultMatrix;        	
    }
    
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

/*
 * http://jersey.java.net/nonav/documentation/snapshot/jaxrs-resources.html
 * 
@POST 
@Path("/postdata3") 
@Consumes("multipart/mixed") 
@Produces("application/json") 
public String postData3(@Multipart(value = "testItem1", type = "application/json") TestItem t1, 
    @Multipart(value = "testItem2", type = "application/json") TestItem t2 
    ); 
§*/
