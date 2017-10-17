/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm;

import java.util.LinkedList;
import java.util.Random;

import weka.core.Instances;

/**
 * Type: ModelInitialisation
 * File: ModelInitialisation.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 *
 */
public abstract class ModelInitialisation
{
	protected final Random rand;
	
	public ModelInitialisation(Random aRand)
	{
		rand = aRand;
	}
	
	
	public abstract Cell generateCell(Instances aInstances);
	
	
	public Cell [] generateCellsArray(Instances aInstances, int numToGenerate)
	{
		Cell [] all = new Cell[numToGenerate];
		for (int i = 0; i < all.length; i++)
		{
			all[i] = generateCell(aInstances);
		}
		return all;
	}
	
	public LinkedList<Cell> generateCellsList(Instances aInstances, int numToGenerate)
	{
		LinkedList<Cell> all = new LinkedList<Cell>();
		for (int i = 0; i < numToGenerate; i++)
		{
			all.add(generateCell(aInstances));
		}
		return all;
	}
}
