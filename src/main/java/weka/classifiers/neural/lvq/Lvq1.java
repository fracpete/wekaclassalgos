package weka.classifiers.neural.lvq;

import java.util.Collection;

import weka.classifiers.Evaluation;
import weka.classifiers.neural.common.learning.LearningKernelFactory;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.algorithm.Lvq1Algorithm;
import weka.core.Instances;


/**
 * Description: Implementation of the LVQ1 algorithm for use in WEKA
 * Implements elements required for the common LVQ algorithm framework
 * specific to the LVQ1 algorithm.
 * 
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 * @author Jason Brownlee
 *
 */
public class Lvq1 extends LvqAlgorithmAncestor
{
	
	

	
	protected void trainModel(Instances instances)
	{
		// construct the algorithm
		LearningRateKernel learningKernel = LearningKernelFactory.factory(learningFunction, learningRate, trainingIterations);
		Lvq1Algorithm algorithm = new Lvq1Algorithm(learningKernel, model, random);
		// add event listeners
		addEventListenersToAlgorithm(algorithm);
		// train the algorithm
		algorithm.trainModel(instances, trainingIterations);
	}
	
	/**
	 * Validate LVQ1 specific arguments
	 * @throws Exception
	 */
    protected void validateArguments() 
    	throws Exception
    {
        // do nothing
    }
	/**
	 * Provide list of LVQ1 specific options
	 * @return Collection
	 */
    protected Collection getListOptions()
    {
        // do nothing
        return null;
    }
	
    protected void setArguments(String [] options)
    	throws Exception
    {}
	/**
	 * Provide collection of LVQ1 specific options
	 * @return Collection
	 */
    protected Collection getAlgorithmOptions()
    {
        // do nothing
        return null;
    }
	/**
	 * Return LVQ1 specific information
	 * @return String
	 */
    public String globalInfo()
    {
		StringBuffer buffer = new StringBuffer(100);
		buffer.append("Learning Vector Quantisation (LVQ) - LVQ1.");
		buffer.append("A single BMU (best matching unit) is selected and moved closer or ");
		buffer.append("further away from each data vector, per iteration.");		
		return buffer.toString();
    }
    /**
     * Entry point into the algorithm for direct usage
     * @param args
     */
    public static void main(String [] args)
    {
        try
        {
            System.out.println(Evaluation.evaluateModel(new Lvq1(), args));
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
        }
    }


}