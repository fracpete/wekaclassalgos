/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm.classification;

import java.util.LinkedList;

import weka.classifiers.immune.airs.algorithm.AISModelClassifier;
import weka.classifiers.immune.airs.algorithm.AffinityFunction;
import weka.classifiers.immune.airs.algorithm.Cell;
import weka.classifiers.immune.airs.algorithm.CellPool;
import weka.core.Instance;
import weka.filters.unsupervised.attribute.Normalize;

/**
 * Type: MajorityVote
 * File: MajorityVote.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 *
 */
public class MajorityVote extends AISModelClassifier
{
	public MajorityVote(
					int aKNumNeighbours,
					Normalize aNormalise,
					CellPool aCellPool,
					AffinityFunction aAffinityFunction)
	{
		super(aKNumNeighbours, aNormalise, aCellPool, aAffinityFunction);
	}
	
	
	protected int classify(Instance aInstance)
	{		
		// respond to affinity
		LinkedList<Cell> cells = model.affinityResponseUnnormalised(aInstance, affinityFunction);
		// determine the majority for the top k cells
		int [] classCounts = determineClassCountForkNN(aInstance, cells);
		
		int largestIndex = 0;
		int largestCount = classCounts[0];
		
		for (int i = 1; i < classCounts.length; i++)
		{
			if(classCounts[i] > largestCount)
			{
				largestCount = classCounts[i];
				largestIndex = i;
			}
		}
		
		return largestIndex;
	}
}
