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

package weka.classifiers.immune.immunos;

import weka.classifiers.immune.affinity.AttributeDistance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

/**
 * Type: Immunos99Algorithm<br>
 * Date: 19/01/2005<br>
 * <br>
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class Immunos99Algorithm implements Serializable {

  public final static NumberFormat format = new DecimalFormat();

  protected Comparator fitnessComparator = new AntibodyFitnessComparator();

  // user paramters
  protected int totalGenerations; // G

  protected long seed; // r

  protected double eta; // E

  protected boolean debug;

  protected double seedPopulationPercentage; // S

  protected LinkedList<Immunos99Antibody> memoryPool;

  protected LinkedList<Immunos99Antibody>[] antibodyGroups;

  protected LinkedList<Immunos99Antibody>[] stock;

  protected Random rand;

  protected DistanceFunction affinityFunction;

  // statistics


  protected double[][] antibodiesPrunedPerGeneration;

  protected double[][] populationSizePerGeneration;

  protected double[][] antibodyFitnessPerGeneration;

  protected double[][] clonesPerGeneration;

  protected double[] totalFinalPrune;

  public Immunos99Algorithm(
    double aEta,
    int aNumGenerations,
    long aSeed,
    double aSeedPercentage,
    boolean aDebug) {
    eta = aEta;
    totalGenerations = aNumGenerations;
    seed = aSeed;
    seedPopulationPercentage = aSeedPercentage;
    debug = aDebug;
  }


  protected void prepareStatistics(int numGroups) {
    if (debug) {
      antibodiesPrunedPerGeneration = new double[numGroups][totalGenerations];
      populationSizePerGeneration = new double[numGroups][totalGenerations];
      antibodyFitnessPerGeneration = new double[numGroups][totalGenerations];
      clonesPerGeneration = new double[numGroups][totalGenerations];
      totalFinalPrune = new double[numGroups];
    }
  }


  protected double sumAffinity(LinkedList<Immunos99Antibody> group, Instance aInstance) {
    double[] dataInstance = aInstance.toDoubleArray();
    double sumAffinity = 0.0;

    for (Immunos99Antibody a : group) {
      double affinity = affinityFunction.distanceEuclideanUnnormalised(a.getAttributes(), dataInstance);
      sumAffinity += affinity;
    }

    return sumAffinity;
  }


  protected double bestAffinity(LinkedList<Immunos99Antibody> group, Instance aInstance) {
    double[] dataInstance = aInstance.toDoubleArray();
    double bestAffinity = Double.POSITIVE_INFINITY;

    for (Immunos99Antibody a : group) {
      double affinity = affinityFunction.distanceEuclideanUnnormalised(a.getAttributes(), dataInstance);
      if (affinity < bestAffinity) {
	bestAffinity = affinity;
      }
    }

    return bestAffinity;
  }

  protected double calculateGroupAvidity(LinkedList<Immunos99Antibody> group, Instance aInstance) {
    double avidity = 0.0;

    // check for empty group
    if (group.isEmpty()) {
      avidity = Double.NaN;
    }
    else {
      // calculate sum affinity
      double affinity = sumAffinity(group, aInstance);
      //    		double affinity = bestAffinity(group, aInstance);

      // protection against divide by zero
      if (affinity == 0) {
	affinity = 1.0;
      }

      avidity = (group.size() / affinity);
    }

    return avidity;
  }

  public double classify(Instance aInstance) {
    // calculate avidity of each group
    double[] avidity = new double[antibodyGroups.length];
    for (int i = 0; i < avidity.length; i++) {
      avidity[i] = calculateGroupAvidity(antibodyGroups[i], aInstance);
    }

    // maximise avidity
    double best = Double.NEGATIVE_INFINITY;
    int bestIndex = -1;
    for (int i = 0; i < avidity.length; i++) {
      // check for an empty group
      if (Double.isNaN(avidity[i])) {
	continue;
      }

      if (avidity[i] > best) {
	best = avidity[i];
	bestIndex = i;
      }
    }

    // check for no classification
    if (bestIndex == -1) {
      return Double.NaN; // don't know the classification
    }

    return bestIndex;
  }

  public double getDataReduction(int totalTrainingInstances) {
    int total = 0;
    for (int i = 0; i < antibodyGroups.length; i++) {
      total += antibodyGroups[i].size();
    }
    double dataReduction = 100.0 * (1.0 - ((double) total / (double) totalTrainingInstances));
    return dataReduction;
  }

  protected String getModelSummary(Instances aInstances) {
    StringBuilder buffer = new StringBuilder();

    int total = 0;
    for (int i = 0; i < antibodyGroups.length; i++) {
      total += antibodyGroups[i].size();
    }

    // data reduction percentage
    double dataReduction = getDataReduction(aInstances.numInstances());

    buffer.append("Data reduction percentage:..." + format.format(dataReduction) + "%\n");
    buffer.append("Total training instances:...." + aInstances.numInstances() + "\n");
    buffer.append("Total cells:................." + total + "\n");
    buffer.append("\n");

    buffer.append(" - Classifier Memory Cells - \n");
    for (int i = 0; i < antibodyGroups.length; i++) {
      buffer.append(aInstances.classAttribute().value(i) + ": " + antibodyGroups[i].size() + "\n");
    }

    return buffer.toString();
  }

  protected String getTrainingSummary(Instances aInstances) {
    StringBuilder b = new StringBuilder();

    if (debug) {
      b.append(" - Training Summary - \n");

      for (int i = 0; i < antibodyGroups.length; i++) {
	if (antibodyGroups[i].isEmpty()) {
	  continue;
	}

	b.append("Group name: " + aInstances.classAttribute().value(i) + "\n");
	b.append("Cells pruned per generation:......" + getStatistic(antibodiesPrunedPerGeneration[i]) + "\n");
	b.append("Population size per generation:..." + getStatistic(populationSizePerGeneration[i]) + "\n");
	b.append("Cell fitness per generation:......" + getStatistic(antibodyFitnessPerGeneration[i]) + "\n");
	b.append("Cloned cells per generation:......" + getStatistic(clonesPerGeneration[i]) + "\n");
	b.append("Cells deleted in final prune:....." + format.format(totalFinalPrune[i]) + "\n");
	b.append("\n");
      }
    }

    b.append("\n");
    b.append(" - Classifier Summary - \n");
    b.append(getModelSummary(aInstances) + "\n");

    return b.toString();
  }


  protected String getStatistic(double[] data) {
    double mean = mean(data);
    double stdev = stdev(data, mean);
    return format.format(mean) + " " + "(" + format.format(stdev) + ")";
  }

  protected double mean(double[] results) {
    double mean = 0.0;
    double sum = 0.0;
    for (int i = 0; i < results.length; i++) {
      sum += results[i];
    }
    mean = (sum / results.length);
    return mean;
  }

  protected double stdev(double[] results, double mean) {
    // standard deviation -
    // square root of the average squared deviation from the mean
    double stdev = 0.0;
    double sum = 0.0;
    for (int i = 0; i < results.length; i++) {
      double diff = mean - results[i];
      sum += diff * diff;
    }
    stdev = Math.sqrt(sum / results.length);
    return stdev;
  }


  protected void algorithmPreperation(Instances aAntigens) {
    // prepare seed
    rand = new Random(seed);
    // distance metric
    affinityFunction = new DistanceFunction(aAntigens);
    // prepare statistics
    prepareStatistics(aAntigens.classAttribute().numValues());
    // initialise antibody set
    initialiseAntibodyPool(aAntigens);
  }

  protected void initialiseAntibodyPool(Instances aAntigens) {
    int total = aAntigens.numInstances();
    total = (int) Math.round(total * seedPopulationPercentage);

    memoryPool = new LinkedList<Immunos99Antibody>();
    aAntigens.randomize(rand);
    for (int i = 0; i < total; i++) {
      Immunos99Antibody antibody = new Immunos99Antibody(aAntigens.instance(i));
      memoryPool.add(antibody);
    }
  }

  protected void train(Instances aInstances)
    throws Exception {
    // prepare the algorithm
    algorithmPreperation(aInstances);
    // sort into antigen-groups
    prepareAntigenGroups(aInstances.numClasses(), aInstances);
    memoryPool = null; // safety

    // for each generation
    for (int gen = 0; gen < totalGenerations; gen++) {
      // for each antigen-group
      for (int group = 0; group < antibodyGroups.length; group++) {
	if (antibodyGroups[group].isEmpty()) {
	  continue;
	}
	// expose the group to all antigens
	clearAccumulatedHistory(antibodyGroups[group]);
	for (int instance = 0; instance < aInstances.numInstances(); instance++) {
	  Instance current = aInstances.instance(instance);
	  updateRankBasedCounts(current, antibodyGroups[group]);
	}
	// fitness
	double[] fitnessValues = calculatePopulationFitness(antibodyGroups[group]);
	// pruning
	int totalPruned = performPruning(eta, antibodyGroups[group], fitnessValues);
	// diversify
	int totalClones = performCloningAndMutation(antibodyGroups[group], stock[group], aInstances, gen);
	insertRandomAntigens(antibodyGroups[group], stock[group], totalPruned, gen);

	if (debug) {
	  antibodiesPrunedPerGeneration[group][gen] = totalPruned;
	  populationSizePerGeneration[group][gen] = antibodyGroups[group].size();
	  antibodyFitnessPerGeneration[group][gen] = mean(fitnessValues);
	  clonesPerGeneration[group][gen] = totalClones;
	}
      }
    }

    // final pruning
    for (int group = 0; group < antibodyGroups.length; group++) {
      if (antibodyGroups[group].isEmpty()) {
	continue;
      }
      // expose the group to all antigens
      clearAccumulatedHistory(antibodyGroups[group]);
      for (int instance = 0; instance < aInstances.numInstances(); instance++) {
	Instance current = aInstances.instance(instance);
	updateClassCount(current, antibodyGroups[group]);
      }
      // calculate fitness
      double[] values = calculatePopulationFitness(antibodyGroups[group]);
      // perform pruning
      int totalPruned = performPruning(eta, antibodyGroups[group], values);
      totalFinalPrune[group] = totalPruned;
    }
  }


  protected void prepareAntigenGroups(int numClasses, Instances aInstances) {
    antibodyGroups = new LinkedList[numClasses];
    stock = new LinkedList[numClasses];
    for (int i = 0; i < antibodyGroups.length; i++) {
      antibodyGroups[i] = new LinkedList<Immunos99Antibody>();
      stock[i] = new LinkedList<Immunos99Antibody>();
    }

    // sort memory pool into groups
    for (Immunos99Antibody a : memoryPool) {
      int classification = (int) a.getClassification();
      antibodyGroups[classification].add(a);
    }

    // stock
    for (int i = 0; i < aInstances.numInstances(); i++) {
      int c = (int) aInstances.instance(i).classValue();
      stock[c].add(new Immunos99Antibody(aInstances.instance(i)));
    }
  }

  protected void clearAccumulatedHistory(LinkedList<Immunos99Antibody> group) {
    for (Immunos99Antibody a : group) {
      a.clearClassCounts();
    }
  }

  protected void insertRandomAntigens(
    LinkedList<Immunos99Antibody> group,
    LinkedList<Immunos99Antibody> stock,
    int totalToIntroduce,
    int generation) {
    totalToIntroduce = Math.min(totalToIntroduce, stock.size());

    // randomise the partition again
    Collections.shuffle(stock, rand);

    // perform insertion
    for (int i = 0; i < totalToIntroduce; i++) {
      // clone the antigen as an antibody
      Immunos99Antibody clone = new Immunos99Antibody(stock.get(i));
      // add to pool
      group.add(clone);
    }
  }


  protected int performCloningAndMutation(
    LinkedList<Immunos99Antibody> group,
    LinkedList<Immunos99Antibody> stock,
    Instances aInstances,
    int generation) {
    LinkedList<Immunos99Antibody> newClones = new LinkedList<Immunos99Antibody>();

    // sort by fitness decending
    Collections.sort(group, fitnessComparator);

    // sum ratios
    double sum = 0.0;
    for (int i = 0; i < group.size(); i++) {
      double ratio = (i + 1) / (double) group.size();
      //            double ratio = ((group.size()-i)/(double)group.size());
      sum += ratio;
    }
    // rank based proliforation and inverse mutation
    for (int i = 0; i < group.size(); i++) {
      Immunos99Antibody antibody = group.get(i);
      double ratio = (i + 1) / (double) group.size();
      //            double ratio = ((group.size()-i)/(double)group.size());
      ratio = (ratio / sum);
      int totalClones = (int) Math.round(ratio * stock.size());
      // generate clones
      for (int j = 0; j < totalClones; j++) {
	// clone
	Immunos99Antibody clone = new Immunos99Antibody(antibody);
	// mutate
	mutateClone(clone, 1.0 - ratio, aInstances); // inverse
	// add to pool
	newClones.add(clone);
      }
    }

    group.addAll(newClones);
    return newClones.size();
  }


  protected class AntibodyFitnessComparator
    implements Comparator<Immunos99Antibody>, Serializable {

    /**
     * Compares its two arguments for order.  Returns a negative integer,
     * zero, or a positive integer as the first argument is less than, equal
     * to, or greater than the second.<p>
     * <p>
     * ascending order
     */
    public int compare(Immunos99Antibody o1, Immunos99Antibody o2) {
      if (o1.getFitness() < o2.getFitness()) {
	return -1;
      }
      else if (o1.getFitness() > o2.getFitness()) {
	return +1;
      }

      return 0;
    }
  }


  protected int performPruning(
    double aEta,
    LinkedList<Immunos99Antibody> group,
    double[] values) {
    double mean;

    if (aEta == -1) {
      mean = mean(values);
      aEta = Math.min(mean, 1.0);
    }

    int count = 0;

    for (Iterator<Immunos99Antibody> iter = group.iterator(); iter.hasNext(); ) {
      Immunos99Antibody a = iter.next();
      double fitness = a.getFitness();

      if (fitness < aEta) {
	iter.remove();
	count++;
      }
    }

    return count;
  }


  protected double[] calculatePopulationFitness(LinkedList<Immunos99Antibody> group) {
    double[] values = new double[group.size()];

    for (int i = 0; i < group.size(); i++) {
      Immunos99Antibody a = group.get(i);

      // calculate fitness
      double fitness = a.calculateFitness();
      values[i] = fitness;
    }

    return values;
  }

  protected void updateRankBasedCounts(Instance aInstance, LinkedList<Immunos99Antibody> group) {
    // calculate affinity for population
    calculateAffinity(group, aInstance);
    // sort by ascending numeric order - best affinity at zero
    Collections.sort(group);
    // allocate rank based scores
    for (int i = 0; i < group.size(); i++) {
      Immunos99Antibody a = group.get(i);
      double score = group.size() - i; // inverse [size,1]
      a.updateClassCount(aInstance, score);
    }
  }

  protected void updateClassCount(Instance aInstance, LinkedList<Immunos99Antibody> group) {
    // calculate affinity for population
    calculateAffinity(group, aInstance);
    // sort by ascending numeric order - best affinity at zero
    Collections.sort(group);
    // retrieve bmu
    Immunos99Antibody bmu = group.getFirst();
    bmu.updateClassCount(aInstance, 1);
  }

  protected void calculateAffinity(
    LinkedList<Immunos99Antibody> antibodies,
    Instance aInstance) {
    double[] data = aInstance.toDoubleArray();

    for (Immunos99Antibody a : antibodies) {
      double affinity = affinityFunction.calculateDistance(a.getAttributes(), data);
      a.setAffinity(affinity);
    }
  }


  protected void mutateClone(
    Antibody aClone,
    double aMutationRate,
    Instances aAntigens
  ) {
    double[][] minmax = affinityFunction.getMinMax();
    AttributeDistance[] attribs = affinityFunction.getDistanceMeasures();

    double[] data = aClone.getAttributes();

    for (int i = 0; i < data.length; i++) {
      if (attribs[i].isClass()) {
	continue;
      }
      else if (attribs[i].isNominal()) {
	// can handel missing values
	data[i] = rand.nextInt(aAntigens.attribute(i).numValues());
      }
      else if (attribs[i].isNumeric()) {
	if (weka.core.Utils.isMissingValue(data[i])) {
	  // select random instance from the stock
	  Immunos99Antibody selected = null;
	  int count = 0;
	  do {
	    int index = (int) aClone.getClassification();
	    int n = rand.nextInt(stock[index].size());
	    selected = stock[index].get(n);
	    if (++count > 10) {
	      break; // use it anyway
	    }
	  }
	  while (!weka.core.Utils.isMissingValue(selected.getAttributes()[i]));
	  data[i] = selected.getAttributes()[i];
	}
	else {
	  // determine the mutation rate based range
	  double range = (minmax[i][1] - minmax[i][0]);
	  range = (range * aMutationRate);

	  // determine bounds for new value based on range
	  double min = Math.max(data[i] - (range / 2.0), minmax[i][0]);
	  double max = Math.min(data[i] + (range / 2.0), minmax[i][1]);

	  // generate new value in VALID range and store
	  data[i] = min + (rand.nextDouble() * (max - min));
	}
      }
      else {
	throw new RuntimeException("Unsuppored attribute type!");
      }
    }
  }

}
