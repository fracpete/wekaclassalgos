package weka.classifiers.neural.lvq.initialise;

import weka.classifiers.neural.common.RandomWrapper;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Date: 25/05/2004
 * File: RandomProportional.java
 * 
 * @author Jason Brownlee
 *
 */
public class RandomProportional extends CommonInitialiser
{	
	public RandomProportional(RandomWrapper aRand, Instances aInstances)
	{
		super(aRand, aInstances);
	}		
	
	public double [] getAttributes()
	{
		// select a random instance			
		int index = makeRandomSelection(totalInstances);
		Instance instance = trainingInstances.instance(index);
		// construct a codebook vector from the selected instance
		double [] attributes = instance.toDoubleArray();
		return attributes;
	}
}
