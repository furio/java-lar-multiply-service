package it.cvdlab.lar.rest;

import it.cvdlab.lar.model.CsrMatrix;

import javax.ws.rs.Consumes;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.MultivaluedMap;
import javax.ws.rs.core.UriInfo;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Path(RestService.REST_SERVICE_URL)
public class RestService {
    private static final Logger logger = LoggerFactory.getLogger(RestService.class);

    public static final String REST_SERVICE_URL = "/multiply";

    // /lar/services/multiply/execute
    @Path("/execute")
    @POST
    @Consumes({ MediaType.APPLICATION_FORM_URLENCODED })
    @Produces({ MediaType.APPLICATION_JSON })
    public CsrMatrix doMultiply(@Context UriInfo uriInfo, MultivaluedMap<String, String> form) {

        return new CsrMatrix();    	
    }
    
    @Path("/test")
    @GET
    @Produces({ MediaType.APPLICATION_JSON })
    public String doTest() {
        return "Che bello";    	
    }    
}
