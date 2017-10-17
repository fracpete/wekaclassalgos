/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.affinity;

import java.io.Serializable;

/**
 * Type: AttributeAffinity
 * File: AttributeAffinity.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 *
 */
public interface AttributeDistance extends Serializable
{
	double distance(double d1, double d2);
	
	boolean isNumeric();
	
	boolean isClass();
	
	boolean isNominal();
}
