package weka.classifiers.neural.lvq.algorithm;

import weka.classifiers.neural.common.RandomWrapper;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.event.EpochEventListener;
import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.classifiers.neural.lvq.model.CommonModel;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;

/**
 * Date: 25/05/2004
 * File: CommonAncestor.java
 * 
 * @author Jason Brownlee
 *
 */
public abstract class CommonAncestor implements Serializable
{	
	protected final LearningRateKernel learningKernel;
	
	protected final CommonModel model;
	
	protected final RandomWrapper rand;
	
	protected final boolean supervised;
	
	protected final LinkedList epochEventListeners;
	
	public CommonAncestor(LearningRateKernel aLearningKernel,						
						  CommonModel aModel,
						  RandomWrapper aRand,
						  boolean isSupervised)
	{
		learningKernel = aLearningKernel;
		model = aModel;
		rand = aRand;
		supervised = isSupervised; 
		epochEventListeners = new LinkedList();
	}
	
	
	public void addEpochEventListener(EpochEventListener aListener)
	{
		epochEventListeners.add(aListener);
	}
	
	protected void activateEpochEventListeners(int aCurrentEpoch, int aTotalEpochs)
	{		
		for (Iterator iter = epochEventListeners.iterator(); iter.hasNext();)
		{
			EpochEventListener element = (EpochEventListener) iter.next();
			element.epochEvent(aCurrentEpoch, aTotalEpochs, model);
		}
	}
	
	
	public abstract void trainModel(Instances aInstances, int numIterations);
	
	
	protected Instance selectRandomInstance(Instances aInstances)	
	{
		int selection = (Math.abs(rand.getRand().nextInt()) % aInstances.numInstances());       
		Instance selectedInstance = aInstances.instance(selection);
		return selectedInstance;
	}

	
	
	protected void updateVector(CodebookVector aCodebookVector, Instance aInstance, double aLearningRate)
	{
		// get attributes for bmu
		double [] attributes = aCodebookVector.getAttributes();

		// determine whether the bmu should be moved closer or further away from the instance
		// if supervised, and the classes of the bmu and data instance do not match
		// then the bmu is moved further away from the data instance
		if(supervised)
		{
			// check if the classification was correct to determine sign of alpha
			if(!isSameClass(aInstance, aCodebookVector))
			{
				aLearningRate = -aLearningRate; // inverse (move away from class)
			}			
		}

		// update all attributes
		for(int i=0; i<attributes.length; i++)
		{
			if (i != aInstance.classIndex() &&  // class attribute cannot be adjusted
				!Utils.isMissingValue(aInstance.value(i))) // never try to adjust towards a missing value
			{
				// calculate the delta (weighted difference) and update codebook vector
				attributes[i] += (aLearningRate * (aInstance.value(i) - attributes[i]));
			}
		}
	}	
	
	
	/**
	 * Determine if the instance and provided codebook vector are in the same class
	 * @param aInstance
	 * @param vector
	 * @return
	 */
	protected boolean isSameClass(Instance aInstance, CodebookVector vector)
	{
		return (vector.getClassification() ==  aInstance.classValue());
	}
	/**
	 * Determine if the two provided codebook vectors are in the same class
	 * @param vector1
	 * @param vector2
	 * @return
	 */
	protected boolean isSameClass(CodebookVector vector1, CodebookVector vector2)
	{
		return (vector1.getClassification() == vector2.getClassification());
	}	
}
