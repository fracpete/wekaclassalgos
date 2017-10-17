package weka.classifiers.neural.lvq.initialise;

import weka.classifiers.neural.common.RandomWrapper;
import weka.core.Instances;

/**
 * Date: 26/05/2004
 * File: RandomValues.java
 * 
 * @author Jason Brownlee
 *
 */
public class RandomValues extends CommonInitialiser
{
	
	public RandomValues(RandomWrapper aRand, Instances aInstances)
	{
		super(aRand, aInstances);		
	}
	
	public double [] getAttributes()
	{
		double [] attributes = new double[numAttributes];
		
		for (int i = 0; i < attributes.length; i++)
		{
			double value = 0.0;
			
			// check for nominal
			if(trainingInstances.attribute(i).isNominal())
			{
				int range = trainingInstances.attribute(i).numValues();
				
				// select a random class value (0 to range-1)
				value = makeRandomSelection(range);
			}
			// generate a random value in the correct range
			else if(trainingInstances.attribute(i).isNumeric())
			{
				double max = trainingInstances.attributeStats(i).numericStats.max;
				double min = trainingInstances.attributeStats(i).numericStats.min;
				// generate a random value in the range of the attribute
				value = ( min + ((max - min) * (rand.getRand().nextDouble() / 1.0)) );	
			}
			
			attributes[i] = value;
		}
		
		return attributes;
	}
}
