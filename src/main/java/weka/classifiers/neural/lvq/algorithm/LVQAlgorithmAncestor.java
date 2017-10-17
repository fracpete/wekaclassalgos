package weka.classifiers.neural.lvq.algorithm;

import weka.classifiers.neural.common.RandomWrapper;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.classifiers.neural.lvq.model.CommonModel;
import weka.core.Instance;
import weka.core.Instances;


/**
 * 
 * Description: Common ancestor for LVQ algorithm implementations,
 * specifically used for building models from training datasets
 * 
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 * @author Jason Brownlee
 *
 */
public abstract class LVQAlgorithmAncestor extends CommonAncestor
{
    
	public LVQAlgorithmAncestor(LearningRateKernel aLearningKernel,						
						  		CommonModel aModel,
						 		RandomWrapper aRand)
	{
		super(aLearningKernel, aModel, aRand, true);
	}
	
	
	
	protected abstract void updateModel(Instance aInstances, double currentLearningRate);
	
	protected abstract boolean usingGlobalLearningRate();
    
    
    
	
    public void trainModel(Instances aInstances, int numIterations)
    {
    	// initialise to the inital learning rate
		double currentLearningRate = learningKernel.getInitialLearningRate(); 
		
        for(int i=0; i<numIterations; i++)
        {
        	// attempt to avoid an unncessary calculation
        	if(usingGlobalLearningRate())
        	{
				// learning rate for this iteration
				currentLearningRate = learningKernel.currentLearningRate(i); 
        	}	       
            // select a random data instance
			Instance selectedInstance = selectRandomInstance(aInstances);
			// update the model using LVQ algorithm
			updateModel(selectedInstance, currentLearningRate);
			// send events
			activateEpochEventListeners(i, numIterations);
        }
    }   
	

  
    /**
     * Used for adjusting individual bmu learning rates consistantly
     * @param aBmu
     * @param aInstance
     */
	protected void adjustIndividualLearningRate(CodebookVector aBmu, Instance aInstance)
	{
		double rate = aBmu.getIndividualLearningRate();
		
		if(isSameClass(aInstance, aBmu))
		{
			// decrease the rate because it was correct
			aBmu.setIndividualLearningRate( rate / (1.0 + rate) );
		}
		else
		{
			// increase the rate because it was incorrect
			aBmu.setIndividualLearningRate( rate / (1.0 - rate) );
			// check for getting two large
			if(aBmu.getIndividualLearningRate() > learningKernel.getInitialLearningRate())
			{
				aBmu.setIndividualLearningRate(learningKernel.getInitialLearningRate());
			}
		}
	}    
}