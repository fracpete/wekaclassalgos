/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.affinity;

/**
 * Type: NumericAttributeDistance
 * File: NumericAttributeDistance.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 *
 */
public class NumericAttributeDistance implements AttributeDistance
{
	/**
	 * @param d1
	 * @param d2
	 * @return
	 */
	public double distance(double d1, double d2)
	{
		double diff = (d1 - d2);
		return (diff * diff);
	}

	
	public boolean isNumeric()
	{
	    return true;
	}	
	public boolean isClass()
	{
	    return false;
	}	
	public boolean isNominal()
	{
	    return false;
	}
}
