package it.cvdlab.lar.utils;

import java.util.Collections;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;

import com.google.common.collect.Lists;

public class FactorList {
	public static int[] getFactors(int nInput)
	{
		int nNumberToFactor = nInput;
		int nCurrentUpper = nInput;
		int i;
	
		List<Integer> factors = Lists.newArrayList();
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
		Collections.sort(factors);
		
		return ArrayUtils.toPrimitive(factors.toArray(new Integer[0]));
	}
}
