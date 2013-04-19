package it.cvdlab.lar.utils;

import java.util.Arrays;


public class PrimeSieve {
	private static final int usefulN = 500000;
	private static boolean[] primeArray;	
	
	static {
		// 0 = false, 1 = false, 2 = true
		primeArray = new boolean[]{false,false,true};
		
		createPrime(usefulN);
	}
	
	private static void createPrime(int N) {
		if (primeArray.length > N) {
			return;
		}
		
		// initially assume all integers are prime
        boolean[] newPrime = new boolean[N + 1];
        Arrays.fill(newPrime, true);
        // 0 and 1 are not primes
        newPrime[0] = false;
        newPrime[1] = false;

        // mark non-primes <= N using Sieve of Eratosthenes
        for (int i = 2; i*i <= N; i++) {

            // if i is prime, then mark multiples of i as nonprime
            // suffices to consider mutiples i, i+1, ..., N/i
            if (newPrime[i]) {
                for (int j = i; i*j <= N; j++) {
                	newPrime[i*j] = false;
                }
            }
        }
        
        primeArray = newPrime;
	}
	
    public static boolean isPrime(int N) { 
    	createPrime(N);

    	return primeArray[N];
    }
}
