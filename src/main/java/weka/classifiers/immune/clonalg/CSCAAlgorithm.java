package weka.classifiers.immune.clonalg;

import weka.classifiers.immune.affinity.AttributeDistance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

/**
 * Type: CLONALGAlgorithm<br>
 * Date: 19/01/2005<br>
 * <br>
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class CSCAAlgorithm implements Serializable {

  public final static NumberFormat format = new DecimalFormat();

  // user paramters
  protected int initialPopulationSize; // S

  protected int totalGenerations; // G

  protected long seed; // r

  protected double alpha; // a

  protected double eta; // E

  protected int kNN; // k

  protected int numPartitions; // p

  protected boolean debug;

  protected LinkedList<CSCAAntibody> memoryPool;

  protected Random rand;

  protected DistanceFunction affinityFunction;

  protected Instances[] partitions;

  protected int partitionIndex;

  // statistics
  protected double[] antibodiesPrunedPerGeneration;

  protected double[] populationSizePerGeneration;

  protected double[] antibodiesWithoutErrorPerGeneration;

  protected double[] antibodyFitnessPerGeneration;

  protected double[] meanAntibodySwitchesPerGeneration;

  protected double[] selectionSetSizePerGeneration;

  protected double[] trainingClassificationAccuracyPerGeneration;

  protected double[] randomInsertionsPerGeneration;

  protected double[] clonesPerGeneration;

  protected int generationsCompleted;

  public CSCAAlgorithm(
    int aInitialPopulationSize,
    int aTotalGenerations,
    long aSeed,
    double aAlpha,
    double aEta,
    int aKNN,
    int aNumPartitions,
    boolean aDebug
  ) {
    initialPopulationSize = aInitialPopulationSize;
    totalGenerations = aTotalGenerations;
    seed = aSeed;
    alpha = aAlpha;
    eta = aEta;
    kNN = aKNN;
    numPartitions = aNumPartitions;
    debug = aDebug;
  }


  protected void prepareStatistics() {
    if (debug) {
      antibodiesPrunedPerGeneration = new double[totalGenerations];
      populationSizePerGeneration = new double[totalGenerations];
      antibodiesWithoutErrorPerGeneration = new double[totalGenerations];
      antibodyFitnessPerGeneration = new double[totalGenerations];
      meanAntibodySwitchesPerGeneration = new double[totalGenerations];
      selectionSetSizePerGeneration = new double[totalGenerations];
      trainingClassificationAccuracyPerGeneration = new double[totalGenerations];
      randomInsertionsPerGeneration = new double[totalGenerations];
      clonesPerGeneration = new double[totalGenerations];
    }
  }

  public double classify(Instance aInstance) {
    // expose the system to the antigen
    CSCAAntibody bmu = selectBestMatchingUnit(aInstance);

    if (kNN == 1) {
      return bmu.getClassification();
    }

    int[] counts = new int[aInstance.classAttribute().numValues()];
    // accumumate counts of for k instances
    for (int i = 0; i < kNN; i++) {
      counts[(int) memoryPool.get(i).getClassification()]++;
    }
    // locate largest
    int bestCount = -1;
    int bestIndex = -1;
    for (int i = 0; i < counts.length; i++) {
      if (counts[i] > bestCount) {
	bestCount = counts[i];
	bestIndex = i;
      }
    }

    return bestIndex;
  }


  protected double classificationAccuracy(Instances aInstances) {
    int correct = 0;

    for (int i = 0; i < aInstances.numInstances(); i++) {
      Instance current = aInstances.instance(i);
      CSCAAntibody bmu = selectBestMatchingUnit(current);
      if (bmu.getClassification() == current.classValue()) {
	correct++;
      }
    }

    return ((double) correct / (double) aInstances.numInstances()) * 100.0;
  }


  protected String getModelSummary(Instances aInstances) {
    StringBuffer buffer = new StringBuffer(1024);

    // data reduction percentage
    double dataReduction = 100.0 * (1.0 - ((double) memoryPool.size() / (double) aInstances.numInstances()));

    buffer.append("Data reduction percentage:..." + format.format(dataReduction) + "%\n");
    buffer.append("Total training instances:...." + aInstances.numInstances() + "\n");
    buffer.append("Total antibodies:............" + memoryPool.size() + "\n");
    buffer.append("\n");

    // determine the breakdown of cells
    int numClasses = aInstances.numClasses();
    int[] counts = new int[numClasses];

    for (CSCAAntibody c : memoryPool) {
      counts[(int) c.getClassification()]++;
    }
    buffer.append(" - Classifier Memory Cells - \n");
    for (int i = 0; i < counts.length; i++) {
      int val = counts[i];
      buffer.append(aInstances.classAttribute().value(i) + ": " + val + "\n");
    }

    return buffer.toString();
  }

  protected String getTrainingSummary(Instances aInstances) {
    StringBuffer b = new StringBuffer(1024);

    if (debug) {
      b.append(" - Training Summary - \n");
      b.append("Generations completed:....................." + generationsCompleted + "\n");
      b.append("Antibodies pruned per generation:.........." + getStatistic(antibodiesPrunedPerGeneration) + "\n");
      b.append("Antibodies without error per generation:..." + getStatistic(antibodiesWithoutErrorPerGeneration) + "\n");
      b.append("Population size per generation:............" + getStatistic(populationSizePerGeneration) + "\n");
      b.append("Antibody fitness per generation:..........." + getStatistic(antibodyFitnessPerGeneration) + "\n");
      b.append("Antibody class switches per generation:...." + getStatistic(meanAntibodySwitchesPerGeneration) + "\n");
      b.append("Selection set size per generation:........." + getStatistic(selectionSetSizePerGeneration) + "\n");
      b.append("Training accuracy per generation:.........." + getStatistic(trainingClassificationAccuracyPerGeneration) + "\n");
      b.append("Inserted antibodies per generation:........" + getStatistic(randomInsertionsPerGeneration) + "\n");
      b.append("Cloned antibodies per generation:.........." + getStatistic(clonesPerGeneration) + "\n");
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
    prepareStatistics();
    // divide dataset into partitions
    preparePartitions(aAntigens);
    // initialise antibody set
    initialiseAntibodyPool(aAntigens);
  }

  protected void initialiseAntibodyPool(Instances aAntigens) {
    // randomise the dataset
    aAntigens.randomize(rand);
    memoryPool = new LinkedList<CSCAAntibody>();
    // select random antigens
    for (int i = 0; i < initialPopulationSize; i++) {
      CSCAAntibody antibody = new CSCAAntibody(aAntigens.instance(i));
      memoryPool.add(antibody);
    }
  }

  protected void preparePartitions(Instances aAntigens) {
    int offset = 0;

    // randomise the dataset
    aAntigens.randomize(rand);
    // determine the number of instances per partition
    int instancesPerPartition = (int) Math.round((double) aAntigens.numInstances() / (double) numPartitions);

    // divide the dataset into partitions
    partitions = new Instances[numPartitions];
    for (int i = 0; i < partitions.length; i++) {
      if (i == partitions.length - 1) {
	// go to the end
	partitions[i] = new Instances(aAntigens, offset, aAntigens.numInstances() - offset);
      }
      else {
	// take a batch
	partitions[i] = new Instances(aAntigens, offset, instancesPerPartition);
	offset += instancesPerPartition;
      }
    }

    // reset index
    partitionIndex = 0;
  }

  protected Instances getNextPartition() {
    Instances partition = partitions[partitionIndex++];
    if (partitionIndex > partitions.length - 1) {
      // loop
      partitionIndex = 0;
    }
    return partition;
  }


  protected void train(Instances aInstances)
    throws Exception {
    boolean stopCondition = false;

    // prepare the algorithm
    algorithmPreperation(aInstances);
    // train the system
    for (int generation = 0; !stopCondition && generation < totalGenerations; generation++) {
      // get a partition
      Instances partition = getNextPartition();
      // randomise the dataset
      partition.randomize(rand);
      // expose the system to the partition
      for (int j = 0; j < partition.numInstances(); j++) {
	// get a data instance
	Instance current = partition.instance(j);
	// locate the best match
	CSCAAntibody bmu = selectBestMatchingUnit(current);
	// accumuate class counts
	bmu.updateClassCount(current);
      }
      // calculate fitness for the population
      calculatePopulationFitness(generation);
      // perform pruning
      int totalPruned = performPruning();
      // prepare the selection set
      LinkedList<CSCAAntibody> selectedSet = prepareSelectedSet();
      if (debug) {
	// statistics
	antibodiesWithoutErrorPerGeneration[generation] = (memoryPool.size() - selectedSet.size());
	selectionSetSizePerGeneration[generation] = selectedSet.size();
      }

      // check for stop condition
      if (selectedSet.isEmpty()) {
	stopCondition = true;
      }
      else {
	// clear accumulated history
	clearAccumulatedHistory();
	// perform cloning and mutation
	performCloningAndMutation(selectedSet, partition, generation);
	// insert random instances
	insertRandomAntigens(partition, selectedSet.size(), generation);
      }

      // statistics
      if (debug) {
	antibodiesPrunedPerGeneration[generation] = totalPruned;
	populationSizePerGeneration[generation] = memoryPool.size();
	trainingClassificationAccuracyPerGeneration[generation] = classificationAccuracy(aInstances);
	generationsCompleted++;
      }
    }
    // perform final pruning
    performFinalPruning();
    //adjust kNN as needed
    kNN = Math.min(kNN, memoryPool.size());
  }

  protected void performFinalPruning() {
    // expose the system to all partitions
    for (int i = 0; i < partitions.length; i++) {
      // get a partition
      Instances partition = partitions[i];
      // randomise the dataset
      partition.randomize(rand);
      // expose the system to the partition
      for (int j = 0; j < partition.numInstances(); j++) {
	// get a data instance
	Instance current = partition.instance(j);
	// locate the best match
	CSCAAntibody bmu = selectBestMatchingUnit(current);
	// accumuate class counts
	bmu.updateClassCount(current);
      }
    }
    // calculate fitness
    calculatePopulationFitness(-1);
    // perform pruning
    performPruning();
  }

  protected void clearAccumulatedHistory() {
    for (CSCAAntibody a : memoryPool) {
      a.clearClassCounts();
    }
  }

  protected void insertRandomAntigens(
    Instances aPartition,
    int totalToIntroduce,
    int generation) {
    totalToIntroduce = Math.min(totalToIntroduce, aPartition.numInstances());

    if (debug) {
      randomInsertionsPerGeneration[generation] = totalToIntroduce;
    }

    // randomise the partition again
    aPartition.randomize(rand);
    // perform insertion
    for (int i = 0; i < totalToIntroduce; i++) {
      // clone the antigen as an antibody
      CSCAAntibody clone = new CSCAAntibody(aPartition.instance(i));
      // add to pool
      memoryPool.add(clone);
    }
  }


  protected void performCloningAndMutation(
    LinkedList<CSCAAntibody> selectedSet,
    Instances aPartition,
    int generation) {
    // determine sum fitness
    double sum = 0.0;
    for (CSCAAntibody a : selectedSet) {
      sum += a.getFitness();
    }
    // perform cloning and mutation
    for (CSCAAntibody a : selectedSet) {
      double ratio = (a.getFitness() / sum);
      int totalClones = (int) Math.round(ratio * (aPartition.numInstances() * alpha));
      // generate clones
      for (int i = 0; i < totalClones; i++) {
	// clone
	CSCAAntibody clone = new CSCAAntibody(a);
	// mutate
	mutateClone(clone, ratio, aPartition);
	// add to pool
	memoryPool.add(clone);
      }

      if (debug) {
	clonesPerGeneration[generation] += totalClones;
      }
    }
  }

  protected LinkedList<CSCAAntibody> prepareSelectedSet() {
    LinkedList<CSCAAntibody> selectedSet = new LinkedList<CSCAAntibody>();
    // add all instance
    selectedSet.addAll(memoryPool);
    // remove all instances without any miss classifications
    for (Iterator<CSCAAntibody> iter = selectedSet.iterator(); iter.hasNext(); ) {
      CSCAAntibody a = iter.next();
      if (!a.hasMisClassified()) {
	// remove from selected set
	iter.remove();
      }
    }

    return selectedSet;
  }


  protected int performPruning() {
    int count = 0;

    for (Iterator<CSCAAntibody> iter = memoryPool.iterator(); iter.hasNext(); ) {
      CSCAAntibody a = iter.next();
      if (a.getFitness() <= eta) {
	iter.remove();
	count++;
      }
    }

    return count;
  }

  protected void calculatePopulationFitness(int generation) {
    for (CSCAAntibody a : memoryPool) {
      // check for a class switch
      if (a.canSwitchClass()) {
	// perform a class switch
	a.switchClasses();

	if (debug && generation != -1) {
	  meanAntibodySwitchesPerGeneration[generation]++;
	}
      }

      // calculate fitness
      double fitness = a.calculateFitness();

      if (debug && generation != -1) {
	antibodyFitnessPerGeneration[generation] += fitness;
      }
    }

    if (debug && generation != -1) {
      antibodyFitnessPerGeneration[generation] /= memoryPool.size();
    }
  }

  protected CSCAAntibody selectBestMatchingUnit(Instance aInstance) {
    CSCAAntibody bmu = null;
    // calculate affinity for population
    calculateAffinity(memoryPool, aInstance);
    // sort by ascending numeric order - best affinity at zero
    Collections.sort(memoryPool);
    // retrieve bmu
    bmu = memoryPool.getFirst();
    return bmu;
  }

  protected void calculateAffinity(
    LinkedList<CSCAAntibody> antibodies,
    Instance aInstance) {
    double[] data = aInstance.toDoubleArray();

    for (CSCAAntibody a : antibodies) {
      double affinity = affinityFunction.calculateDistance(a.getAttributes(), data);
      a.setAffinity(affinity);
    }
  }


  protected void mutateClone(
    Antibody aClone,
    double aMutationRate,
    Instances aPartition) {
    double[][] minmax = affinityFunction.getMinMax();
    AttributeDistance[] attribs = affinityFunction.getDistanceMeasures();

    double[] data = aClone.getAttributes();

    for (int i = 0; i < data.length; i++) {
      if (attribs[i].isClass()) {
	continue;
      }
      else if (attribs[i].isNominal()) {
	data[i] = rand.nextInt(aPartition.attribute(i).numValues());
      }
      else if (attribs[i].isNumeric()) {
	// determine the mutation rate based range
	double range = (minmax[i][1] - minmax[i][0]);
	range = (range * aMutationRate);

	// determine bounds for new value based on range
	double min = Math.max(data[i] - (range / 2.0), minmax[i][0]);
	double max = Math.min(data[i] + (range / 2.0), minmax[i][1]);

	// generate new value in VALID range and store
	data[i] = min + (rand.nextDouble() * (max - min));
      }
      else {
	throw new RuntimeException("Unsuppored attribute type!");
      }
    }
  }
}
