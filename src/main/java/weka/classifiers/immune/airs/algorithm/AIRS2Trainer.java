/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm;

import weka.classifiers.immune.airs.algorithm.classification.MajorityVote;
import weka.classifiers.immune.airs.algorithm.initialisation.RandomInstancesInitialisation;
import weka.classifiers.immune.airs.algorithm.samplegeneration.StimulationProportionalMutation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.text.NumberFormat;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

/**
 * Type: AIRS1Trainer
 * File: AIRS1Trainer.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class AIRS2Trainer implements AISTrainer {

  protected final double affinityThresholdScalar;

  protected final double clonalRate;

  protected final double hyperMutationRate;

  protected final double totalResources;

  protected final double stimulationThreshold;

  protected final int affinityThresholdNumInstances;

  protected final Random rand;

  protected final int memoryCellPoolInitialSize;

  protected final int kNN;

  protected AffinityFunction affinityFunction;

  protected SampleGenerator arbSampleGeneration;

  protected double affinityThreshold;

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


  public AIRS2Trainer(
    double aAffinityThresholdScalar,
    double aClonalRate,
    double aHyperMutationRate,
    double aTotalResources,
    double aStimulationValue,
    int aNumInstancesAffinityThreshold,
    Random aRand,
    int aMemoryCellPoolInitialSize,
    int aKNN) {
    affinityThresholdScalar = aAffinityThresholdScalar;
    clonalRate = aClonalRate;
    hyperMutationRate = aHyperMutationRate;
    totalResources = aTotalResources;
    stimulationThreshold = aStimulationValue;
    affinityThresholdNumInstances = aNumInstancesAffinityThreshold;
    rand = aRand;
    memoryCellPoolInitialSize = aMemoryCellPoolInitialSize;
    kNN = aKNN;
  }


  public void algorithmPreperation(Instances aInstances) {
    affinityFunction = new AffinityFunction(aInstances);
    arbSampleGeneration = prepareSampleGeneration(aInstances);
  }

  protected SampleGenerator prepareSampleGeneration(Instances aInstances) {
    return new StimulationProportionalMutation(rand);
  }

  public String getTrainingSummary() {
    StringBuilder buffer = new StringBuilder();
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
    buffer.append("Mean ARB prunings per refinement iteration:......" + f.format((double) totalArbDeletions / (double) totalArbRefinementIterations) + "\n");

    return buffer.toString();
  }


  public AISModelClassifier train(Instances instances)
    throws Exception {
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

  public void setAffinityThreshold(double a) {
    affinityThreshold = a;
  }

  protected AISModelClassifier internalTrain(
    Instances trainingSet,
    Normalize normalise)
    throws Exception {
    // initialise model
    initialise(trainingSet);

    // train model on each instance
    for (int i = 0; i < trainingSet.numInstances(); i++) {
      Instance current = trainingSet.instance(i);
      CellPool arbCellPool = new CellPool(new LinkedList<Cell>());

      // identify best match from memory pool
      Cell bestMatch = identifyMemoryPoolBestMatch(current);
      if (bestMatch == null) {
	bestMatch = addNewMemoryCell(current);
      }
      // check for an identical match
      else if (bestMatch.getStimulation() == 1.0) {
	// do nothing
      }
      else {
	// generate arbs and add to arb pool
	generateARBs(arbCellPool, bestMatch, current);
	// perform ARB refinement
	Cell candidate = runARBRefinement(arbCellPool, current);
	// respond to candidate
	respondToCandidateMemoryCell(bestMatch, candidate, current);
      }

      //			System.out.println("Finished "+(i+1)+"/"+trainingSet.numInstances());
    }

    // prepare statistics
    prepareStatistics(trainingSet.numInstances());
    // prepare the classifier
    AISModelClassifier classifier = getClassifier(normalise);
    return classifier;
  }

  protected void prepareStatistics(int aNumTrainingInstances) {
    totalTrainingInstances = aNumTrainingInstances;
    meanClonesArb /= totalArbRefinementIterations;
    meanClonesMemCell /= totalTrainingInstances;
    meanAllocatedResources /= totalArbRefinementIterations;
    meanArbPoolSize /= totalArbRefinementIterations;
    meanArbRefinementIterations = ((double) totalArbRefinementIterations / (double) totalTrainingInstances);

  }

  protected Cell runARBRefinement(
    CellPool aArbCellPool,
    Instance aInstance) {
    boolean stopCondition = false;
    Cell candidateMemoryCell = null;

    do {
      // perform competition for resources
      candidateMemoryCell = performARBCompetitionForResources(aArbCellPool, aInstance);
      // calculate if stop condition has been met
      stopCondition = isStoppingCriterion(aArbCellPool, aInstance);

      if (!stopCondition) {
	LinkedList<Cell> arbs = new LinkedList<Cell>();
	// 3c. variation (mutated clones)
	for (Cell c : aArbCellPool.getCells()) {
	  arbs.addAll(generateARBVarients(aInstance, c));
	}
	aArbCellPool.add(arbs);
      }

      // stats
      meanArbPoolSize += aArbCellPool.size();
      meanArbRefinementIterations++;
      totalArbRefinementIterations++;
    }
    while (!stopCondition);

    return candidateMemoryCell;
  }


  protected AISModelClassifier getClassifier(Normalize aNormalise) {
    MajorityVote classifier = new MajorityVote(
      kNN,
      aNormalise,
      memoryCellPool,
      affinityFunction);
    return classifier;
  }

  protected void respondToCandidateMemoryCell(
    Cell bestMatchMemoryCell,
    Cell candidateMemoryCell,
    Instance aInstance) {
    // recalculate candidate stimulation
    double candidateStimulation = stimulation(candidateMemoryCell, aInstance);
    // check if candidate is better
    if (candidateStimulation > bestMatchMemoryCell.getStimulation()) {
      // add candidate to memory pool
      memoryCellPool.add(candidateMemoryCell);
      // check previous best can be removed
      double affinity = affinityFunction.affinityNormalised(bestMatchMemoryCell, candidateMemoryCell);
      if (affinity < getMemoryCellReplacementCutoff()) {
	// remove previous best
	memoryCellPool.delete(bestMatchMemoryCell);
	totalMemoryCellReplacements++;
      }
    }
  }


  protected LinkedList<Cell> generateARBVarients(Instance aInstance, Cell aArb) {
    LinkedList<Cell> newARBs = new LinkedList<Cell>();

    // determine the number of clones to produce
    int numClones = arbNumClones(aArb);
    // generate clones
    for (int i = 0; i < numClones; i++) {
      // generate mutated clone
      Cell mutatedClone = arbSampleGeneration.generateSample(aArb, aInstance);

      // add to arb pool
      newARBs.add(mutatedClone);
    }

    meanClonesArb += numClones;

    return newARBs;
  }


  protected boolean isStoppingCriterion(
    CellPool aArbCellPool,
    Instance aInstance) {
    double meanStimulation = 0.0;

    // sum stimulation values
    for (Iterator<Cell> iter = aArbCellPool.iterator(); iter.hasNext(); ) {
      Cell c = iter.next();
      meanStimulation += c.getStimulation();
    }

    meanStimulation = (meanStimulation / aArbCellPool.size());

    // check if the stopping condition has been met
    // that is the mean is >= the stimulation threshold
    if (meanStimulation >= stimulationThreshold) {
      return true;
    }

    // safety
    if (Double.isNaN(meanStimulation)) {
      throw new RuntimeException("Infinite loop condition detected, mean stimulation is NaN.");
    }

    // condition is not met
    return false;
  }


  protected Cell performARBCompetitionForResources(
    CellPool aArbCellPool,
    Instance aInstance) {
    Cell mostStimulatedSameClass = null;
    double numResAllowed = totalResources;

    // stimulate arbs, normalise stimulation, order by stimulation
    stimulationNormalisation(aArbCellPool, aInstance);
    // allocate resources to arbs based on stimulation
    double resources = calculateResourceAllocations(aArbCellPool, aInstance);

    // continue until the resources for this class is below a threshold
    LinkedList<Cell> cells = aArbCellPool.getCells();
    while (resources > numResAllowed) {
      double numResourceToRemove = (resources - numResAllowed);
      Cell last = cells.getLast();

      // check if element can be removed
      if (last.getNumResources() <= numResourceToRemove) {
	// remove from everywhere
	cells.removeLast();
	totalArbDeletions++;
	resources -= last.getNumResources();
      }
      else {
	// decrement resources
	double res = last.getNumResources() - numResourceToRemove;
	last.setNumResources(res);
	resources -= numResourceToRemove;
      }
    }

    // best ARB will always have the most resources
    mostStimulatedSameClass = cells.getFirst();

    // stats
    meanAllocatedResources += resources;
    return mostStimulatedSameClass;
  }


  protected double calculateResourceAllocations(
    CellPool cellPool,
    Instance aInstance) {
    double resources = 0.0;

    for (Iterator<Cell> iter = cellPool.iterator(); iter.hasNext(); ) {
      Cell c = iter.next();
      double r = (c.getStimulation() * clonalRate);
      c.setNumResources(r);
      resources += r;
    }
    // order by allocated resources
    cellPool.orderByResources();
    return resources;
  }

  protected void generateARBs(
    CellPool arbCellPool,
    Cell aBestMatchMemoryCell,
    Instance aInstance) {
    // add best match to the arb pool
    arbCellPool.add(new Cell(aBestMatchMemoryCell));

    // determine the number of clones to produce
    int numClones = memoryCellNumClones(aBestMatchMemoryCell);
    // generate clones
    for (int i = 0; i < numClones; i++) {
      // generate mutated clone
      Cell mutatedClone = arbSampleGeneration.generateSample(aBestMatchMemoryCell, aInstance);
      // add to arb pool
      arbCellPool.add(mutatedClone);
    }

    meanClonesMemCell += numClones;
  }


  protected Cell identifyMemoryPoolBestMatch(Instance aInstance) {
    // get memory pool sorted by stimulation
    LinkedList<Cell> stimulatedSorted = stimulation(memoryCellPool.getCells(), aInstance);
    // process list until a member of the same class is located
    for (Cell c : stimulatedSorted) {
      if (Utils.isSameClass(aInstance, c)) {
	return c;
      }
    }

    return null;
  }

  protected Cell addNewMemoryCell(Instance aInstance) {
    // no match, therefore create one
    Cell c = new Cell(aInstance);
    // add to memory cell pool
    memoryCellPool.add(c);
    double s = stimulation(c, aInstance);
    c.setStimulation(s);
    return c;
  }


  protected void initialise(Instances aTrainingSet) {
    ModelInitialisation init = getModelInitialisation();
    memoryCellPool = new CellPool(init.generateCellsList(aTrainingSet, memoryCellPoolInitialSize));
  }


  protected ModelInitialisation getModelInitialisation() {
    return new RandomInstancesInitialisation(rand);
  }

  /**
   * The number of clones that an ARB can produce
   *
   * @param aArb
   * @return
   */
  protected int arbNumClones(Cell aArb) {
    return (int) Math.round(aArb.getStimulation() * clonalRate);
  }

  /**
   * The numberof clones that a memory cell can produce
   *
   * @param aArb
   * @return
   */
  protected int memoryCellNumClones(Cell aArb) {
    return (int) Math.round(aArb.getStimulation() * clonalRate * hyperMutationRate);
  }


  protected double getMemoryCellReplacementCutoff() {
    return (affinityThreshold * affinityThresholdScalar);
  }


  protected void stimulationNormalisation(
    CellPool cells,
    Instance aInstance) {
    double min = Double.POSITIVE_INFINITY;
    double max = Double.NEGATIVE_INFINITY;

    // determine min and max
    for (Iterator<Cell> iter = cells.iterator(); iter.hasNext(); ) {
      Cell c = iter.next();
      double s = stimulation(c, aInstance);

      if (s < min) {
	min = s;
      }
      if (s > max) {
	max = s;
      }
    }

    // normalise
    double range = (max - min);

    if (range == 0) {
      throw new RuntimeException("Infinite loop condition detected: range of stimulation values is zero.");
    }

    for (Iterator<Cell> iter = cells.iterator(); iter.hasNext(); ) {
      Cell c = iter.next();
      double normalised = (c.getStimulation() - min) / range;
      c.setStimulation(normalised);

      // validation
      if (normalised < 0 || normalised > 1) {
	throw new RuntimeException("Normalised stimulation outside range!");
      }
    }
  }

  protected LinkedList<Cell> stimulation(LinkedList<Cell> cells, Instance aInstance) {
    // calculate stimulation for all the cells
    for (Cell c : cells) {
      stimulation(c, aInstance);
    }
    // order the population by stimulation
    Collections.sort(cells, CellPool.stimulationComparator);
    return cells;
  }

  protected double stimulation(Cell aCell, Instance aInstance) {
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
