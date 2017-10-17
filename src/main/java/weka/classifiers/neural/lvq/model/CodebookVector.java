package weka.classifiers.neural.lvq.model;

import java.io.Serializable;

import weka.core.Instance;

/**
 * 
 * Description: Represents a single codebook vector in an LVQ model
 * A codebook vector is also called a prototype or an exemplar. It is
 * a single node which represents a sign post in the state space
 * of the training data
 * 
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 * @author Jason Brownlee
 *
 */
public class CodebookVector 
	implements Serializable
{
	protected final int codebookId;
	
	/**
	 * Vecotrs attribute values
	 */
    protected double [] attributes;
	/**
	 * Index of the class attribute
	 */
    protected int classAttributeIndex;
	/**
	 * Distance from a specific data instance at a moment in time
	 */
    protected double distance;
	/**
	 * Used in cases where codebook vectors can have their own individual learning rate
	 */
    protected double learningRate;
    /**
     * The number of times that the codebook vector is the bmu
     */
    protected int bmuCorrectCount;
    
	protected int bmuIncorrectCount;
	
	private boolean voting;
	
	private int [] classHitDistribution;   
    
    
    
    
	public CodebookVector(int aCodebookId)
	{
		codebookId = aCodebookId;
	}	
	
	public void setClassification(double aClassificationValue)
	{
		attributes[classAttributeIndex] = aClassificationValue;
	}
	
	public void initialise(double [] aAttributes, int aClassIndex, int aNumClasses)
	{
		attributes = aAttributes;
		classAttributeIndex = aClassIndex;
		classHitDistribution = new int[aNumClasses];
	}
	
	
    
	/**
	 * String representation of this codebook vector
	 * @return String
	 */
    public String toString()
    {
        StringBuffer buffer = new StringBuffer(100);
        
        // all attributes
        for(int i=0; i<attributes.length; i++)
        {
            buffer.append(attributes[i]);
            buffer.append(", ");
        }
        
        // class index
		buffer.append(getClassification());

        return buffer.toString();
    }
    /**
     * Return codebook vectors class assignmnet
     * @return
     */
    public double getClassification()
    {
    	if(voting)
    	{
    		int largestIndex = 0;
    		
    		// find index with largest value
    		for (int i = 1; i < classHitDistribution.length; i++)
			{
				if(classHitDistribution[i] > classHitDistribution[largestIndex])		
				{
					largestIndex = i;		
				}
			}
			
			// check for all empty
			if(classHitDistribution[largestIndex] == 0)
			{
				return attributes[classAttributeIndex];
			}
			else
			{
				return (double) largestIndex;
			}
    	}
    	
        return attributes[classAttributeIndex];
    }
	/**
	 * Get distance from data instance at a point in time
	 * @return
	 */
    public double getDistance()
    {
        return distance;
    }
	/**
	 * Return codebook vector's internal representation
	 * @return
	 */
    public double [] getAttributes()
    {
        return attributes;
    }
	/**
	 * codebook vectors individual learning rate
	 * @return
	 */
    public double getIndividualLearningRate()
    {
        return learningRate;
    }
    /**
     * Set the codebook vectors learning rate
     * @param lrate
     */
    public void setIndividualLearningRate(double lrate)
    {
        learningRate = lrate;
    }
    
    public double value(int aIndex)
    {
    	return attributes[aIndex];
    }
    
    public void setBmuHit(double aDistance, Instance aInstance)
    {
		distance = aDistance;
		
		// check for training mode
		if(!Instance.isMissingValue(aInstance.classValue()))
		{
			if(aInstance.classValue() == getClassification())
			{
				bmuCorrectCount++;
			}
			else
			{
				bmuIncorrectCount++;		
			}
			
			// store for class distribution
			classHitDistribution[(int)aInstance.classValue()]++;			
		}
    }
    
    public void resetBmuCounts()
    {
		bmuCorrectCount = bmuIncorrectCount = 0;
    }
    public int getBmuCorrectCount()
    {
    	return bmuCorrectCount;
    }
	public int getBmuIncorrectCount()
	{
		return bmuIncorrectCount;
	}
	
	public int getId()
	{
		return codebookId;
	}
	
	public int [] getClassHitDistribution()
	{
		return classHitDistribution;
	}
	
	public boolean hasClassChanged()
	{
		return (getClassification() != attributes[classAttributeIndex]);
	}
	public void setUseVoting(boolean useVoting)
	{
		voting = useVoting;
	}
	public void clearClassDistributions()
	{
		for (int i = 0; i < classHitDistribution.length; i++)
		{
			classHitDistribution[i] = 0;
		}
	}
}