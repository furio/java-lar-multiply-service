package it.cvdlab.lar.clengine;

import java.util.List;

import org.bridj.Pointer;

import com.google.common.collect.Lists;

final class PonterUtils {
	static <T> void copyToPointer(List<T> iList, Pointer<T> oPointer) {
		for(int i = 0; i < iList.size(); i++) {
			oPointer.set(i, iList.get(i));
		}
	}
	
	/*
	static <T extends Number> List<T> copyFromPointer(Pointer<T> lPointer) {
		List<T> tmpList = Lists.newArrayList();
		for(T singleData: lPointer) {
			tmpList.add( singleData );
		}
		return Lists.newArrayList(tmpList);
	}
	*/

	static List<Integer> copyFromPointerInteger(Pointer<Integer> iPointer) {
		List<Integer> tmpList = Lists.newArrayList();
		for(Integer singleData: iPointer) {
			tmpList.add( new Integer(singleData) );
		}
		return Lists.newArrayList(tmpList);
	}
	
	static List<Float> copyFromPointerFloat(Pointer<Float> fPointer) {
		List<Float> tmpList = Lists.newArrayList();
		for(Float singleData: fPointer) {
			tmpList.add( new Float(singleData) );
		}
		return Lists.newArrayList(tmpList);
	}
}
