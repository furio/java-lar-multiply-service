package it.cvdlab.lar.clengine;

final class KernelConfig {
	private static final String KERNEL_DENSE_LOCAL = "SpMSpM-Multiply-Naive.cl";
	private static final String KERNEL_DENSE_NOLOCAL = "SpMSpM-Multiply-Naive.nl.cl";
	private static final String KERNEL_COO_LOCAL = "SpMSpM-Multiply-COO.cl";
	private static final String KERNEL_COO_NOLOCAL = "SpMSpM-Multiply-COO.nl.cl";
	private static final String KERNEL_NNZ_LOCAL = "NNZ-Calc.cl";
	private static final String KERNEL_NNZ_NOLOCAL = "NNZ-Calc.nl.cl";
	
	public static final String KERNEL_DENSE() {
		if (CLEngineConfig.isIMPL_LOCAL()) {
			return KERNEL_DENSE_NOLOCAL;
		} else {
			return KERNEL_DENSE_LOCAL;
		}
	}
	
	public static final String KERNEL_COO() {
		if (CLEngineConfig.isIMPL_LOCAL()) {
			return KERNEL_COO_NOLOCAL;
		} else {
			return KERNEL_COO_LOCAL;
		}
	}
	
	public static final String KERNEL_NNZ() {
		if (CLEngineConfig.isIMPL_LOCAL()) {
			return KERNEL_NNZ_NOLOCAL;
		} else {
			return KERNEL_NNZ_LOCAL;
		}
	}	
	
	public static final String KERNEL_DENSE_FUN_FULL = "spmm_kernel_naive";
	public static final String KERNEL_DENSE_FUN_SHORT = "spmm_binary_kernel_naive";
	public static final String KERNEL_COO_FUN_FULL = "spmm_coo_kernel_naive";
	public static final String KERNEL_COO_FUN_SHORT = "spmm_coo_binary_kernel_naive";
	public static final String KERNEL_NNZ_FUN = "nnz_calc_kernel";
	
	public static final String DEFINE_ROW = "%%AROW%%";
	public static final String DEFINE_COL = "%%BCOL%%";
}
