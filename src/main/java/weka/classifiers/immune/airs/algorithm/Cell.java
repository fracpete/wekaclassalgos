/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm;

import java.io.Serializable;

import weka.core.Instance;

/**
 * Type: Cell
 * File: Cell.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 *
 */
public class Cell implements Serializable
{
    private final double [] attributes;
	
    private final int classIndex;
	
	private long usage;
	
	/**
	 * 
	 */
	protected double affinity;
	
	/**
	 * number of resources held by the cell
	 */
	protected double numResources;
	
	/**
	 * current stimulation value
	 */
	protected double stimulation;	
	
	
	
	
	
	public Cell(double [] aAttributes, int aClassIndex)
	{
		attributes = aAttributes;
		classIndex = aClassIndex;
	}
	
	public Cell(Instance aInstance)
	{
		// note to double array creates a new object
		this(aInstance.toDoubleArray(), aInstance.classIndex());
	}
	
	
	public Cell(Cell aCell)
	{
		classIndex = aCell.classIndex;
		attributes = new double[aCell.attributes.length];
		System.arraycopy(aCell.attributes, 0, attributes, 0, attributes.length);
	}
	
	
	
	public double getClassification()
	{
		return attributes[classIndex];
	}
	
	public double [] getAttributes()
	{
		return attributes;
	}
	
	public int getClassIndex()
	{
		return classIndex;
	}
	
	
	public double getAffinity()
	{
		return affinity;
	}
	public void setAffinity(double affinity)
	{
		this.affinity = affinity;
	}
	
	
	
	public double getNumResources()
	{
		return numResources;
	}
	public void setNumResources(double numResources)
	{
		this.numResources = numResources;
	}
	public double getStimulation()
	{
		return stimulation;
	}
	public void setStimulation(double stimulation)
	{
		this.stimulation = stimulation;
	}
	
	protected long getUsage()
	{
	    return usage;
	}
	protected void incrementUsage()
	{
	    usage++;
	}
	protected void clearUsage()
	{
	    usage = 0;
	}
}
