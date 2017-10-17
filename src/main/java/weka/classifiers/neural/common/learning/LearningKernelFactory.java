package weka.classifiers.neural.common.learning;

import weka.core.Tag;

/**
 * Date: 25/05/2004
 * File: LearningRateFactory.java
 * 
 * @author Jason Brownlee
 *
 */
public class LearningKernelFactory
{
	public final static int LEARNING_FUNCTION_LINEAR  = 1;
	public final static int LEARNING_FUNCTION_INVERSE = 2;
	public final static int LEARNING_FUNCTION_STATIC  = 3;
	
	public final static Tag [] TAGS_LEARNING_FUNCTION =
	{
		new Tag(LEARNING_FUNCTION_LINEAR,  "Linear Decay"),
		new Tag(LEARNING_FUNCTION_INVERSE, "Inverse"),
		new Tag(LEARNING_FUNCTION_STATIC,  "Static")
	};	
	
	public final static String [] LEARNING_FUNCTION_FULL_DESC =
	{
	   "Linear decay learning rate function",
	   "Inverse learning rate function",
	   "Static learning rate"
	};
	
	public static String getDescription(int aLearningFunction)
	{
		return LEARNING_FUNCTION_FULL_DESC[aLearningFunction-1];
	}
	
	public final static String DESCRIPTION;
	
	static
	{
		StringBuffer buffer = new StringBuffer();
		buffer.append("(");		
		
		for (int i = 0; i < TAGS_LEARNING_FUNCTION.length; i++)
		{
			buffer.append(TAGS_LEARNING_FUNCTION[i].getID());
			buffer.append("==");
			buffer.append(TAGS_LEARNING_FUNCTION[i].getReadable());			
			
			if(i != TAGS_LEARNING_FUNCTION.length-1)
			{
				buffer.append(", ");
			}
		}
		buffer.append(")");
		
		DESCRIPTION = buffer.toString();
	}
	
	public final static LearningRateKernel factory(int aLearningRate, double initalLearningRate, int totalIterations)	
	{
		LearningRateKernel kernel = null;
		
		switch(aLearningRate)
		{
			case LEARNING_FUNCTION_LINEAR:
			{
				kernel = new LinearLearningRate(initalLearningRate, totalIterations);
				break;
			}
			case LEARNING_FUNCTION_INVERSE:
			{
				kernel = new InverseLearningRate(initalLearningRate, totalIterations);
				break;
			}		
			case LEARNING_FUNCTION_STATIC:
			{
				kernel = new StaticLearningRate(initalLearningRate, totalIterations);
				break;
			}		
			default:
			{
				throw new RuntimeException("Unknown learning rate: " + aLearningRate);
			}
		}
		
		return kernel;
	}
}
