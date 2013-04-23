package it.cvdlab.lar.utils;

import java.util.Collections;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

public class FactorList {
	public static int[] getFactors(int nInput) {
		return getFactorsLong(nInput);
	}
	
	@SuppressWarnings("unused")
	private static int[] getFactorsShort(int nInput) {
		int nNumberToFactor = nInput;
		int nCurrentUpper = nInput;
		int i;
	
		List<Integer> factors = Lists.newArrayList();
		// 1 always is a factor
		factors.add(1);
	
		for (i = 2; i < nCurrentUpper; i++) {
			if ((nNumberToFactor % i) == 0) {
				// if we found a factor, the upper number is the new upper limit
				nCurrentUpper = nNumberToFactor / i;
				factors.add(i);
			
				if (nCurrentUpper != i) // avoid "double counting" the square root
					factors.add(nCurrentUpper);
			}
		}
		// myself is always a factor
		factors.add(nInput);
		// order list
		Collections.sort(factors);
		
		return ArrayUtils.toPrimitive(factors.toArray(new Integer[0]));
	}

	private static int[] getFactorsLong(int nInput) {
		if (nInput < 1) {
			return new int[]{};
		}
		
		List<Integer> small = Lists.newArrayList();
		List<Integer> large = Lists.newArrayList();
		int end = (int) Math.floor(Math.sqrt(nInput));
		
		for (int i = 1; i <= end; i++) {
			if ((nInput % i) == 0) {
				small.add(i);
				if ((i * i) != nInput) {  // Don't include a square root twice
					large.add(nInput / i);
				}
			}
		}
		
		return ArrayUtils.toPrimitive( Lists.newArrayList(
		        Iterables.concat(small, Lists.reverse(large))).toArray(new Integer[0]) );
	}
}
