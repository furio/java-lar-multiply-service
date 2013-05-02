package it.cvdlab.lar.clengine;

import it.cvdlab.lar.utils.FactorList;
import it.cvdlab.lar.utils.PrimeSieve;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Predicate;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.primitives.Ints;

public class SizeEstimator {
	private static final Logger logger = LoggerFactory.getLogger(SizeEstimator.class);
	
	private static final int MINIMUM_DIVISORS = 2;
	private static final int MAXIMUM_TRIES = 20;
	
	private static int[] getGoodSingleSize(final int bound, final int maxBound, int minimumDivisors, int maximumTries) throws Exception {
		// To filter divisor list
		Predicate<Integer> currFilter = new Predicate<Integer>() {
	        @Override
	        public boolean apply(Integer input) {
	        	return ((input > 1) && (input < maxBound));
	        }
	    };
	    
	    //
		int newX = bound;
		int divX = 0;
		int i = 0;
		
		// If we can get something decent in the next "maximumTries"
		while (i <= maximumTries) {
			newX += i;

			if ( !PrimeSieve.isPrime(newX) ) {
				List<Integer> factorList = Lists.newArrayList( 
						Iterables.filter( 
								Ints.asList( FactorList.getFactors(newX) ), 
								currFilter ) );

				logger.debug(newX + "-" + factorList.size() + "-" + factorList);
				// System.out.println(newX + "-" + factorList.size() + "-" + factorList);
				if ( factorList.size() >= minimumDivisors ) {
					factorList = Lists.reverse(factorList);
					logger.debug(newX + "- L: " + factorList);
					divX = factorList.get(0);
					return new int[]{newX, divX};
				}
			}

			i += 1;
		}
		
		throw new Exception("Cannot find a suitable couple of divisors!");
	}	

	private static int[] getGoodSingleSize(int bound, int maxBound) throws Exception {
		return getGoodSingleSize(bound, maxBound, MINIMUM_DIVISORS, MAXIMUM_TRIES);
	}
	
	public static List<int[]> getGoodSizes(int sizeX, int sizeY, int boundSize) throws Exception {
		int maxSqrt = (int) Math.ceil( Math.sqrt(boundSize) );
		logger.info("Calculating sizes ... [" + sizeX + "," + sizeY + "] Bound: [" + boundSize + "," + maxSqrt + "]");
		
		if ( (sizeX*sizeY) < boundSize ) {
			return Lists.newArrayList(new int[]{sizeX, sizeY}, new int[]{sizeX, sizeY});
		}
		
		int[] newX = getGoodSingleSize(sizeX, maxSqrt);
		int[] newY = getGoodSingleSize(sizeY, maxSqrt);
		
		return Lists.newArrayList(new int[]{newX[0], newY[0]}, new int[]{newX[1], newY[1]});
	}
}
