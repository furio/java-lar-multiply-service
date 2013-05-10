package it.cvdlab.lar.clengine;

final class KernelConfig {
	public static final String KERNEL_DENSE = "SpMSpM-Multiply-Naive.cl";
	public static final String KERNEL_COO = "SpMSpM-Multiply-COO.cl";
	public static final String KERNEL_NNZ = "NNZ-Calc.cl";
	
	public static final String KERNEL_DENSE_FUN_FULL = "spmm_kernel_naive";
	public static final String KERNEL_DENSE_FUN_SHORT = "spmm_binary_kernel_naive";
	public static final String KERNEL_COO_FUN_FULL = "spmm_coo_kernel_naive";
	public static final String KERNEL_COO_FUN_SHORT = "spmm_coo_binary_kernel_naive";
	public static final String KERNEL_NNZ_FUN = "nnz_calc_kernel";
	
	public static final String DEFINE_ROW = "%%AROW%%";
	public static final String DEFINE_COL = "%%BCOL%%";
}
