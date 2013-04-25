package it.cvdlab.lar.rest;

import java.io.IOException;

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

import it.cvdlab.lar.model.CsrMatrix;
import it.cvdlab.lar.clengine.MultiplyCL;

@Path(RestService.REST_SERVICE_URL)
public class RestService {
	private static final String MATRIX_FIRST_PARAM = "matrixA";
	private static final String MATRIX_SECOND_PARAM = "matrixB";
	
	
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
    	
    	CsrMatrix firstMatrix = null;
    	CsrMatrix secondMatrix = null;
    	boolean firstParse = false;
    	boolean secondParse = false;
    	
    	System.err.println(form.toString());
    	
    	if ( form.containsKey(MATRIX_FIRST_PARAM) ) {
    		try {
    			firstMatrix = jacksonMapper.readValue(form.getFirst(MATRIX_FIRST_PARAM), CsrMatrix.class);
    			firstParse = true;
			} catch (JsonParseException e) {
				System.err.println( e.toString() );
			} catch (JsonMappingException e) {
				System.err.println( e.toString() );
			} catch (IOException e) {
				System.err.println( e.toString() );
			}
    	}
    	
    	if ( form.containsKey(MATRIX_SECOND_PARAM) ) {
    		try {
    			secondMatrix = jacksonMapper.readValue(form.getFirst(MATRIX_SECOND_PARAM), CsrMatrix.class);
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
    		resultMatrix = MultiplyCL.multiply(firstMatrix, secondMatrix);
    	}
    	
    	
    	logger.error("/execute");
        return resultMatrix;    	
    }
    
    @Path("/networktest")
    @GET
    @Produces({ MediaType.APPLICATION_JSON })
    public String doTest() {
    	// httpServletResponse.addHeader("Access-Control-Allow-Origin", "*");

        return (new CsrMatrix(new int[]{0,1,2}, new int[]{0,1}, 2, 2)).toDense().toString();    	
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
