/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm.initialisation;

import java.util.Random;

import weka.classifiers.immune.airs.algorithm.Cell;
import weka.classifiers.immune.airs.algorithm.ModelInitialisation;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Type: RandomInstancesInitialisation
 * File: RandomInstancesInitialisation.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 *
 */
public class RandomInstancesInitialisation extends ModelInitialisation
{
	
	public RandomInstancesInitialisation(Random aRand)
	{
		super(aRand);
	}

	/**
	 * @param aInstances
	 * @return
	 */
	public Cell generateCell(Instances aInstances)
	{
		int selection = rand.nextInt(aInstances.numInstances());
		Instance inst = aInstances.instance(selection);
		return new Cell(inst);
	}

}
