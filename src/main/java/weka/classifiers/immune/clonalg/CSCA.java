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

package weka.classifiers.immune.clonalg;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.UnsupportedClassTypeException;
import weka.core.Utils;

import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Vector;

/**
 * Type: CLONALG <br>
 * Date: 19/01/2005 <br>
 * <br>
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class CSCA extends AbstractClassifier {

  // user paramters
  protected int initialPopulationSize; // S

  protected int totalGenerations; // G

  protected long seed; // r

  protected double clonalScaleFactor; // a

  protected double minimumFitnessThreshold; // E

  protected int kNN; // k

  protected int numPartitions; // p

  protected CSCAAlgorithm algorithm;

  protected String trainingSummary;


  private final static String[] PARAMETERS =
    {
      "S",
      "G",
      "r",
      "a",
      "E",
      "k",
      "p"
    };

  private final static String[] DESCRIPTIONS =
    {
      "Initial population size (S).",
      "Total generations (G).",
      "Random number generator seed (r).",
      "Clonal scale factor (Alpha).",
      "Minimum fitness threshold (Eta).",
      "k-Nearest Neighbours (k).",
      "Total Partitions (p)."
    };


  public CSCA() {
    // set defaults
    initialPopulationSize = 50;
    totalGenerations = 5;
    seed = 1;
    clonalScaleFactor = 1.0;
    minimumFitnessThreshold = 1.0;
    kNN = 1;
    numPartitions = 1;

    // TODO: should not be true by default
    m_Debug = true;
  }


  public void buildClassifier(Instances data) throws Exception {
    Instances trainingInstances = new Instances(data);

    // must have a class assigned
    if (trainingInstances.classIndex() < 0) {
      throw new Exception("No class attribute assigned to instances");
    }
    // class must be nominal
    else if (!trainingInstances.classAttribute().isNominal()) {
      throw new UnsupportedClassTypeException("Class attribute must be nominal");
    }
    // must have attributes besides the class attribute
    else if (trainingInstances.numAttributes() <= +1) {
      throw new Exception("Dataset contains no supported comparable attributes");
    }

    // delete with missing class
    trainingInstances.deleteWithMissingClass();
    for (int i = 0; i < trainingInstances.numAttributes(); i++) {
      trainingInstances.deleteWithMissing(i);
    }

    // must have some training instances
    if (trainingInstances.numInstances() == 0) {
      throw new Exception("No usable training instances!");
    }

    // validation
    performParameterValidation(trainingInstances);

    // construct trainer
    algorithm = new CSCAAlgorithm(
      initialPopulationSize,
      totalGenerations,
      seed,
      clonalScaleFactor,
      minimumFitnessThreshold,
      kNN,
      numPartitions,
      m_Debug
    );

    // train
    algorithm.train(trainingInstances);
    // training summary
    trainingSummary = algorithm.getTrainingSummary(trainingInstances);
  }

  protected void performParameterValidation(Instances trainingInstances)
    throws Exception {
    if (numPartitions >= trainingInstances.numInstances()) {
      throw new Exception("Total partitions is more than or equal to the number of training instances.");
    }
    if (numPartitions <= 0) {
      throw new Exception("Total partitions must be > 0 and < total training instances.");
    }

    if (initialPopulationSize > trainingInstances.numInstances()) {
      throw new Exception("The initial population size is larger than the number of training instances.");
    }
    if (initialPopulationSize <= 0) {
      throw new Exception("Initial population size must be > 0 and <= total training instances.");
    }

  }

  public double classifyInstance(Instance instance) throws Exception {
    if (algorithm == null) {
      throw new Exception("Algorithm has not been prepared.");
    }
    return algorithm.classify(instance);
  }

  public String toString() {
    StringBuffer buffer = new StringBuffer(1000);
    buffer.append("Clonal Selection Classification Algorithm (CSCA) v1.0.\n");

    if (trainingSummary != null) {
      buffer.append("\n");
      buffer.append(trainingSummary);
    }

    return buffer.toString();
  }

  public String globalInfo() {
    StringBuffer buffer = new StringBuffer(1000);
    buffer.append(toString());
    buffer.append("\n\n");

    buffer.append("Jason Brownlee.  " +
      "[Technical Report].  " +
      "Clonal Selection Theory & CLONAG - The Clonal Selection Classification Algorithm (CSCA).  " +
      "Victoria, Australia: Centre for Intelligent Systems and Complex Processes (CISCP), " +
      "Faculty of Information and Communication Technologies (ICT), " +
      "Swinburne University of Technology; " +
      "2005 Jan; " +
      "Technical Report ID: 2-01.\n");
    buffer.append("\\n");
    buffer.append("http://www.it.swin.edu.au/centres/ciscp/ais/\n");


    return buffer.toString();
  }

  public Enumeration listOptions() {
    Vector<Option> list = new Vector<Option>(15);

    // add parents options
    Enumeration e = super.listOptions();
    while (e.hasMoreElements()) {
      list.add((Option) e.nextElement());
    }

    // add new options
    for (int i = 0; i < PARAMETERS.length; i++) {
      Option o = new Option(DESCRIPTIONS[i], PARAMETERS[i], 1, "-" + PARAMETERS[i]);
      list.add(o);
    }

    return list.elements();
  }


  protected double getDouble(String param, String[] options) throws Exception {
    String value = Utils.getOption(param.charAt(0), options);
    if (value == null) {
      throw new Exception("Parameter not provided: " + param);
    }

    return Double.parseDouble(value);
  }

  protected int getInteger(String param, String[] options) throws Exception {
    String value = Utils.getOption(param.charAt(0), options);
    if (value == null) {
      throw new Exception("Parameter not provided: " + param);
    }

    return Integer.parseInt(value);
  }

  protected long getLong(String param, String[] options) throws Exception {
    String value = Utils.getOption(param.charAt(0), options);
    if (value == null) {
      throw new Exception("Parameter not provided: " + param);
    }

    return Long.parseLong(value);
  }

  public void setOptions(String[] options) throws Exception {
    // parental option setting
    super.setOptions(options);

    setInitialPopulationSize(getInteger(PARAMETERS[0], options));
    setTotalGenerations(getInteger(PARAMETERS[1], options));
    setSeed(getLong(PARAMETERS[2], options));
    setClonalScaleFactor(getDouble(PARAMETERS[3], options));
    setMinimumFitnessThreshold(getDouble(PARAMETERS[4], options));
    setKNN(getInteger(PARAMETERS[5], options));
    setNumPartitions(getInteger(PARAMETERS[6], options));
  }


  public String[] getOptions() {
    LinkedList<String> list = new LinkedList<String>();

    String[] options = super.getOptions();
    for (int i = 0; i < options.length; i++) {
      list.add(options[i]);
    }

    list.add("-" + PARAMETERS[0]);
    list.add(Integer.toString(initialPopulationSize));
    list.add("-" + PARAMETERS[1]);
    list.add(Integer.toString(totalGenerations));
    list.add("-" + PARAMETERS[2]);
    list.add(Long.toString(seed));
    list.add("-" + PARAMETERS[3]);
    list.add(Double.toString(clonalScaleFactor));
    list.add("-" + PARAMETERS[4]);
    list.add(Double.toString(minimumFitnessThreshold));
    list.add("-" + PARAMETERS[5]);
    list.add(Integer.toString(kNN));
    list.add("-" + PARAMETERS[6]);
    list.add(Integer.toString(numPartitions));

    return list.toArray(new String[list.size()]);
  }


  public String initialPopulationSizeTipText() {
    return DESCRIPTIONS[0];
  }

  public String totalGenerationsTipText() {
    return DESCRIPTIONS[1];
  }

  public String seedTipText() {
    return DESCRIPTIONS[2];
  }

  public String clonalScaleFactorTipText() {
    return DESCRIPTIONS[3];
  }

  public String minimumFitnessThresholdTipText() {
    return DESCRIPTIONS[4];
  }

  public String kNNTipText() {
    return DESCRIPTIONS[5];
  }

  public String numPartitionsTipText() {
    return DESCRIPTIONS[6];
  }


  public double getClonalScaleFactor() {
    return clonalScaleFactor;
  }

  public void setClonalScaleFactor(double clonalScaleFactor) {
    this.clonalScaleFactor = clonalScaleFactor;
  }

  public int getInitialPopulationSize() {
    return initialPopulationSize;
  }

  public void setInitialPopulationSize(int initialPopulationSize) {
    this.initialPopulationSize = initialPopulationSize;
  }

  public int getKNN() {
    return kNN;
  }

  public void setKNN(int knn) {
    kNN = knn;
  }

  public double getMinimumFitnessThreshold() {
    return minimumFitnessThreshold;
  }

  public void setMinimumFitnessThreshold(double minimumFitnessThreshold) {
    this.minimumFitnessThreshold = minimumFitnessThreshold;
  }

  public int getNumPartitions() {
    return numPartitions;
  }

  public void setNumPartitions(int numPartitions) {
    this.numPartitions = numPartitions;
  }

  public long getSeed() {
    return seed;
  }

  public void setSeed(long seed) {
    this.seed = seed;
  }

  public int getTotalGenerations() {
    return totalGenerations;
  }

  public void setTotalGenerations(int totalGenerations) {
    this.totalGenerations = totalGenerations;
  }

  public static void main(String[] argv) {

    try {
      System.out.println(Evaluation.evaluateModel(new CSCA(), argv));
    }
    catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }
}
