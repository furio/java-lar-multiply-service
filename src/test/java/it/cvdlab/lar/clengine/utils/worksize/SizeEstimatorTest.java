package it.cvdlab.lar.clengine.utils.worksize;

import static org.junit.Assert.*;
import it.cvdlab.lar.clengine.utils.worksize.SizeEstimator;
import it.cvdlab.lar.clengine.utils.worksize.SizeEstimatorException;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class SizeEstimatorTest {
	
	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void testGetGoodSizes() {
		try {
			SizeEstimator.getGoodSizes(0, 0, 10);
			fail("Should throw exception");
		} catch(SizeEstimatorException e) {
			
		}
	}

}
