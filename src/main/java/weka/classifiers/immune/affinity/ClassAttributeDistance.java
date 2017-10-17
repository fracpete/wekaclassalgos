/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.affinity;

/**
 * Type: ClassAttributeDistance
 * File: ClassAttributeDistance.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 *
 */
public class ClassAttributeDistance implements AttributeDistance
{

	/**
	 * @param d1
	 * @param d2
	 * @return
	 */
	public double distance(double d1, double d2)
	{
		return 0.0; // never compare classes
	}
	
	public boolean isNumeric()
	{
	    return false;
	}	
	public boolean isClass()
	{
	    return true;
	}	
	public boolean isNominal()
	{
	    return true;
	}

}
