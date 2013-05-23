package it.cvdlab.lar.clengine.utils.worksize;

public enum SizeEnum {
	SINGLE (1),
    EIGHT   (8),
    SIXTEEN   (16),
    THIRTYTWO (32),
    HUNDREDTWENTYEIGHT (128);
    
	private int vectorsize;
	SizeEnum(int vs) { vectorsize = vs; }
	public int getVectorsize() {
		return vectorsize;
	}
}
