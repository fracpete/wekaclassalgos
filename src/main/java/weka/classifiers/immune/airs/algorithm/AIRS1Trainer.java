/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm;

import java.text.NumberFormat;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

import weka.classifiers.immune.airs.algorithm.classification.MajorityVote;
import weka.classifiers.immune.airs.algorithm.initialisation.RandomInstancesInitialisation;
import weka.classifiers.immune.airs.algorithm.samplegeneration.RandomMutate;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 * Type: AIRS1Trainer
 * File: AIRS1Trainer.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 *
 */
public class AIRS1Trainer implements AISTrainer
{    
	protected final double affinityThresholdScalar;	
	protected final double clonalRate;	
	protected final double hyperMutationRate;
	protected final double mutationRate;
	protected final double totalResources;	
	protected final double stimulationThreshold;	
	protected final int affinityThresholdNumInstances; 
	protected final Random rand;
	protected final int arbCellPoolInitialSize;
	protected final int memoryCellPoolInitialSize;	
	protected final int kNN;	
	
	protected AffinityFunction affinityFunction;	
	protected SampleGenerator arbSampleGeneration;
	
	protected double affinityThreshold;
	
	protected CellPool arbMemoryCellPool;
	protected CellPool memoryCellPool;	
	
	// stats
	protected double meanClonesArb;
	protected double meanClonesMemCell;
	protected double meanAllocatedResources;
	protected double meanArbPoolSize;	
	protected double meanArbRefinementIterations;	
	protected long totalArbDeletions;
	protected long totalMemoryCellReplacements;
	protected long totalArbRefinementIterations;
	protected long totalTrainingInstances;
	
	
	public AIRS1Trainer(
					double aAffinityThresholdScalar,
					double aClonalRate,
					double aHyperMutationRate,
					double aMutationRate,
					double aTotalResources,
					double aStimulationValue,
					int aNumInstancesAffinityThreshold,
					Random aRand,
					int aArbCellPoolInitialSize,
					int aMemoryCellPoolInitialSize,
					int aKNN)
	{
		affinityThresholdScalar = aAffinityThresholdScalar;
		clonalRate = aClonalRate;
		hyperMutationRate = aHyperMutationRate;
		mutationRate = aMutationRate;
		totalResources = aTotalResources;
		stimulationThreshold = aStimulationValue;
		affinityThresholdNumInstances = aNumInstancesAffinityThreshold;
		rand = aRand;
		arbCellPoolInitialSize = aArbCellPoolInitialSize;
		memoryCellPoolInitialSize = aMemoryCellPoolInitialSize;
		kNN = aKNN;
	}
	
	
	public void algorithmPreperation(Instances aInstances)
	{
		affinityFunction = new AffinityFunction(aInstances);
		arbSampleGeneration = prepareSampleGeneration(aInstances);
	}
	
	protected SampleGenerator prepareSampleGeneration(Instances aInstances)
	{
		return new RandomMutate(rand, aInstances.numClasses(), mutationRate);
	}
	
	protected void log(String s)
	{
	    System.out.println(s);
	}
	
	public String getTrainingSummary()
	{
	    StringBuffer buffer = new StringBuffer(1024);
	    NumberFormat f = Utils.format;
	    
	    buffer.append(" - Training Summary - \n");
	    buffer.append("Affinity Threshold:.............................." + f.format(affinityThreshold) + "\n");	    
	    buffer.append("Total training instances:........................" + f.format(totalTrainingInstances) + "\n");
	    buffer.append("Total memory cell replacements:.................." + f.format(totalMemoryCellReplacements) + "\n");	    
	    buffer.append("Mean ARB clones per refinement iteration:........" + f.format(meanClonesArb) + "\n");
	    buffer.append("Mean total resources per refinement iteration:..." + f.format(meanAllocatedResources) + "\n");
	    buffer.append("Mean pool size per refinement iteration:........." + f.format(meanArbPoolSize) + "\n");	    
	    buffer.append("Mean memory cell clones per antigen:............." + f.format(meanClonesMemCell) + "\n");
	    buffer.append("Mean ARB refinement iterations per antigen:......" + f.format(meanArbRefinementIterations) + "\n");	    
	    buffer.append("Mean ARB prunings per refinement iteration:......" + f.format((double)totalArbDeletions/(double)totalArbRefinementIterations) + "\n");
	    	    	    
	    return buffer.toString();
	}
	
	
	public AISModelClassifier train(Instances instances)
	throws Exception
	{
		// normalise the dataset
		Normalize normalise = new Normalize();
		normalise.setInputFormat(instances);
		Instances trainingSet = Filter.useFilter(instances, normalise);
		// prepare the algorithm
		algorithmPreperation(trainingSet);		
		// calculate affinity threshold
		affinityThreshold = Utils.calculateAffinityThreshold(trainingSet, affinityThresholdNumInstances, rand, affinityFunction);
		// perform the training
		return internalTrain(trainingSet, normalise);
	}
	
	
	public void setAffinityThreshold(double a)
	{
		affinityThreshold = a;
	}
	
	
	protected AISModelClassifier internalTrain(
					Instances trainingSet, 
					Normalize normalise)
		throws Exception
	{
		// initialise model
		initialise(trainingSet);
		
		// train model on each instance
		for (int i = 0; i < trainingSet.numInstances(); i++)
		{
			Instance current = trainingSet.instance(i);
			
			// identify best match from memory pool
			Cell bestMatch = identifyMemoryPoolBestMatch(current);
			if(bestMatch == null)
			{
			    bestMatch = addNewMemoryCell(current);
			}
			// never process best match that is identical to the instance
			else if(bestMatch.getStimulation() == 1.0)
			{
				// do nothing
			}
			else
			{
				// generate arbs and add to arb pool
				generateARBs(bestMatch, current);
				// get the candidate memory cell
				Cell candidateMemoryCell = runARBRefinement(current);
				// introduce the memory cell
				respondToCandidateMemoryCell(bestMatch, candidateMemoryCell, current);
			}
		}
		
		// prepare statistics
		prepareStatistics(trainingSet.numInstances());		
		// prepare the classifier
		AISModelClassifier classifier = getClassifier(normalise);
		return classifier;
	}	
	
	
	
	protected void prepareStatistics(int aNumTrainingInstances)
	{
		totalTrainingInstances = aNumTrainingInstances;
		meanClonesArb /= totalArbRefinementIterations;
		meanClonesMemCell /= totalTrainingInstances;
		meanAllocatedResources /= totalArbRefinementIterations;
		meanArbPoolSize /= totalArbRefinementIterations;	
		meanArbRefinementIterations = ((double)totalArbRefinementIterations / (double)totalTrainingInstances);

	}
	
	protected Cell runARBRefinement(Instance aInstance)	
	{
		boolean stopCondition = false;
		boolean firstTime = true;
		Cell candidateMemoryCell = null;
		
		do
		{
			// perform competition for resources
			candidateMemoryCell = performARBCompetitionForResources(aInstance);			
			// calculate if stop condition has been met
			stopCondition = isStoppingCriterion(aInstance);				
			
			// always executed the first time, or when the stop condition is not met
			if(!stopCondition || firstTime)
			{
				LinkedList<Cell> arbs = new LinkedList<Cell>();				
				// 3c. variation (mutated clones)
				for(Cell c : arbMemoryCellPool.getCells())
				{
				    arbs.addAll(generateARBVarients(aInstance, c));						
				}
				arbMemoryCellPool.add(arbs);
				firstTime = false;
			}
			
			// stats
			meanArbPoolSize += arbMemoryCellPool.size();
			meanArbRefinementIterations++;
			totalArbRefinementIterations++;
		}
		while(!stopCondition);
		
		return candidateMemoryCell;
	}
	
	
	
	
	
	
	protected AISModelClassifier getClassifier(Normalize aNormalise)
	{
		MajorityVote classifier = new MajorityVote(kNN, aNormalise, memoryCellPool, affinityFunction);
		return classifier;
	}
	
	protected void respondToCandidateMemoryCell(
					Cell bestMatchMemoryCell, 
					Cell candidateMemoryCell,
					Instance aInstance)
	{
		// recalculate candidate stimulation
		double candidateStimulation = stimulation(candidateMemoryCell, aInstance);
		// check if candidate is better
		if(candidateStimulation > bestMatchMemoryCell.getStimulation())
		{
			// add candidate to memory pool
			memoryCellPool.add(candidateMemoryCell);
			// check previous best can be removed
			double affinity = affinityFunction.affinityNormalised(bestMatchMemoryCell, candidateMemoryCell);
			if(affinity < getMemoryCellReplacementCutoff())
			{
				// remove previous best
				memoryCellPool.delete(bestMatchMemoryCell);
				totalMemoryCellReplacements++;
			}			
		}
	}
	
	
	protected LinkedList<Cell> generateARBVarients(Instance aInstance, Cell aArb)
	{
		LinkedList<Cell> newARBs = new LinkedList<Cell>();
		
		// determine the number of clones to produce
		int numClones = arbNumClones(aArb);
		// generate clones
		for (int i = 0; i < numClones; i++)
		{
			// generate mutated clone
			Cell mutatedClone = arbSampleGeneration.generateSample(aArb, aInstance);
			
			// add to arb pool
			newARBs.add(mutatedClone);
		}
		
		meanClonesArb += numClones;
		
		return newARBs;
	}
	
	
	
	protected boolean isStoppingCriterion(Instance aInstance)
	{
		// calculate the mean stimulation level for each class
		int numClasses = aInstance.numClasses();
		double []  meanStimulation = new double[numClasses];
		double [] classCount = new double[numClasses];
		
		for(Cell c : arbMemoryCellPool.getCells())
		{
			int index = (int) c.getClassification();
			meanStimulation[index] += c.getStimulation();
			classCount[index]++;
		}
		
		// calculate means - all means must be >= stimulation threshold
		for (int i = 0; i < meanStimulation.length; i++)
		{
			meanStimulation[i] = (meanStimulation[i] / classCount[i]);
			if(meanStimulation[i] < stimulationThreshold)
			{
				return false;
			}
		}
		
		return true;
	}
	
	protected double determineMaximumResourceAllocation(
					Instance aInstance,
					int aClassIndex, 
					int aNumClasses)
	{
		double numResAllowed = 0.0;
		
		if(aClassIndex == aInstance.classValue())
		{
			numResAllowed = totalResources / 2.0;
		}
		else
		{
			numResAllowed = totalResources / (2.0 * (aNumClasses-1));
		}
		
		return numResAllowed;
	}
	
	
	protected LinkedList<Cell> getAllArbsInClass(int aClassValue)
	{
		LinkedList<Cell> cells = new LinkedList<Cell>();
		
		for (Iterator<Cell> iter = arbMemoryCellPool.iterator(); iter.hasNext();)
		{
			Cell c = iter.next();
			if(aClassValue == c.getClassification())
			{
				cells.add(c);
			}
		}
		
		return cells;
	}
	
	protected Cell performARBCompetitionForResources(Instance aInstance)
	{
		Cell mostStimulatedSameClass = null;
		
		// calculate stimulation levels
		LinkedList<Cell> sortedStimulated = stimulationNormalisation(arbMemoryCellPool.getCells(),aInstance);
		// normalise stimulation, allocate resources, sum resources for each class
		double [] resources = calculateResourceAllocations(sortedStimulated, aInstance);
		
		// perform resource management;
		for (int i = 0; i < resources.length; i++)
		{
			// calculate resources allowed
			double numResAllowed = determineMaximumResourceAllocation(aInstance, i, resources.length);			
			// collect all ARBs in this class
			LinkedList<Cell> cells = getAllArbsInClass(i);			
			// sort by resource
			Collections.sort(cells, CellPool.resourceComparator);
			
			// continue until the resources for this class is below a threshold
			while(resources[i] > numResAllowed)
			{
				double numResourceToRemove = (resources[i]-numResAllowed);
				
				Cell last = cells.getLast();
				// check if element can be removed
				if(last.getNumResources() <= numResourceToRemove)
				{
					cells.removeLast(); // remove from the temp list
					arbMemoryCellPool.delete(last); // remove from the ARB pool
					totalArbDeletions++;
					resources[i] -= last.getNumResources();
				}
				else
				{
					// decrement resources
					double res = last.getNumResources() - numResourceToRemove;
					last.setNumResources(res);
					resources[i] -= numResourceToRemove;
				}
			}
			
			// special case of same class as training instance
			if(i == aInstance.classValue())
			{
				// the list is orded by resource allocations, thus the best
				// cell is always at the beginning of the list
				mostStimulatedSameClass = cells.getFirst();
			}
		}
		
		for (int i = 0; i < resources.length; i++)
        {
		    meanAllocatedResources += resources[i];
        }
		
		return mostStimulatedSameClass;
	}
	
	
	protected double [] calculateResourceAllocations(
					LinkedList<Cell> list, 
					Instance aInstance)
	{		
	    double [] resources = new double[aInstance.numClasses()];
		
		for(Cell c : list)
		{			
			// check for not the same class
			if(!Utils.isSameClass(aInstance, c))
			{
				double s = (1.0 - c.getStimulation()); // invert
				c.setStimulation(s);
			}			
			
			double resource = c.getStimulation() * clonalRate;
			c.setNumResources(resource);
			// sum resources
			resources[(int)c.getClassification()] += resource;
		}
		
		return resources;
	}
	
	protected void generateARBs(Cell aBestMatchMemoryCell, Instance aInstance)
	{
		// add best match to the arb pool
		arbMemoryCellPool.add(new Cell(aBestMatchMemoryCell));
		
		// determine the number of clones to produce
		int numClones = memoryCellNumClones(aBestMatchMemoryCell);
		// generate clones
		for (int i = 0; i < numClones; i++)
		{
			// generate mutated clone
			Cell mutatedClone = arbSampleGeneration.generateSample(aBestMatchMemoryCell,aInstance);
			
			// add to arb pool
			arbMemoryCellPool.add(mutatedClone);
		}
		
		meanClonesMemCell += numClones;
	}
	
	
	protected Cell identifyMemoryPoolBestMatch(Instance aInstance)
	{
		// get memory pool sorted by stimulation
		LinkedList<Cell> stimulatedSorted = stimulation(memoryCellPool.getCells(),aInstance);
		// process list until a member of the same class is located
		for(Cell c : stimulatedSorted)
		{
			if(Utils.isSameClass(aInstance, c))
			{
				return c;
			}
		}
		
		return null;
	}
	
	protected Cell addNewMemoryCell(Instance aInstance)
	{
		// no match, therefore create one
		Cell c = new Cell(aInstance);
		// add to memory cell pool
		memoryCellPool.add(c);
		return c;
	}
	
	
	protected void initialise(Instances aTrainingSet)
	{		
		ModelInitialisation init = getModelInitialisation();
		arbMemoryCellPool = new CellPool(init.generateCellsList(aTrainingSet, arbCellPoolInitialSize));
		memoryCellPool = new CellPool(init.generateCellsList(aTrainingSet, memoryCellPoolInitialSize));
	}	
	
	
	protected ModelInitialisation getModelInitialisation()
	{
		return new RandomInstancesInitialisation(rand);
	}
	
	/**
	 * The number of clones that an ARB can produce
	 * @param aArb
	 * @return
	 */
	protected int arbNumClones(Cell aArb)
	{
		return (int) Math.round(aArb.getStimulation() * clonalRate);
	}
	
	/**
	 * The numberof clones that a memory cell can produce
	 * @param aArb
	 * @return
	 */
	protected int memoryCellNumClones(Cell aArb)
	{
		return (int) Math.round(aArb.getStimulation() * clonalRate * hyperMutationRate);
	}
	
	
	
	protected double getMemoryCellReplacementCutoff()
	{
		return (affinityThreshold * affinityThresholdScalar);
	}
	
	
	protected LinkedList<Cell> stimulationNormalisation(
	        LinkedList<Cell> cells, 
	        Instance aInstance)
	{
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
	    
		// determine min and max
		for(Cell c : cells)
		{
			double s = stimulation(c, aInstance);
			
			if(s < min)
			{
				min = s;
			}
			if(s > max)
			{
				max = s;
			}
		}
		
		// normalise
		double range = (max - min);
		for(Cell c : cells)
		{
		    double s = c.getStimulation();
		    double normalised = (s-min) / range;
		    c.setStimulation(normalised);
		    
		    // validation
		    if(normalised<0 || normalised>1)
		    {
		        throw new RuntimeException("Normalised stimulation outside range!");
		    }
		}
	    
	    return cells;
	}
	
	protected LinkedList<Cell> stimulation(LinkedList<Cell> cells, Instance aInstance)
	{
	    // calculate stimulation for all the cells
	    for(Cell c : cells)
	    {
	        stimulation(c, aInstance);
	    }
	    // order the population by stimulation
	    Collections.sort(cells, CellPool.stimulationComparator);
	    return cells;
	}
	
	protected double stimulation(Cell aCell, Instance aInstance)
	{
	    // calculate normalised affinity [0,1]
	    double affinity = affinityFunction.affinityNormalised(aInstance, aCell);
	    // convert to stimulation
	    double stimulation = 1.0 - affinity;
	    // store
	    aCell.setStimulation(stimulation);
	    // return it in case its needed
	    return stimulation;
	}
}
