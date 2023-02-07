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

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHelper;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Normalize;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

/**
 * Type: Immunos99 <br>
 * Date: 19/01/2005 <br>
 * <br>
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class Immunos99 extends AbstractClassifier {

  // user paramters
  protected int totalGenerations; // G

  protected long seed; // r

  protected double minimumFitnessThreshold; // E

  protected double seedPopulationPercentage; //S

  protected Immunos99Algorithm algorithm;

  protected String trainingSummary;

  protected Normalize normaliser;

  protected int totalTrainingInstances;

  private final static String[] PARAMETERS =
    {
      "G",
      "r",
      "E",
      "S"
    };

  private final static String[] DESCRIPTIONS =
    {
      "Total generations (G).",
      "Random number generator seed (r).",
      "Minimum fitness threshold (Eta).",
      "Seed population percentage (S)."
    };


  public Immunos99() {
    // set defaults
    totalGenerations = 1;
    seed = 1;
    minimumFitnessThreshold = -1;
    seedPopulationPercentage = 0.2;

    // TODO: should not be true by default
    m_Debug = true;
  }

  /**
   * Returns the Capabilities of this classifier.
   *
   * @return the capabilities of this object
   * @see Capabilities
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    result.disableAll();

    // attributes
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.NOMINAL_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    result.setMinimumNumberInstances(1);

    return result;
  }

  public void buildClassifier(Instances data) throws Exception {
    Instances trainingInstances = new Instances(data);
    trainingInstances.deleteWithMissingClass();

    getCapabilities().testWithFail(trainingInstances);

    // validation
    performParameterValidation(trainingInstances);

    // construct trainer
    algorithm = new Immunos99Algorithm(
      minimumFitnessThreshold,
      totalGenerations,
      seed,
      seedPopulationPercentage,
      m_Debug);

    // normalise the dataset
    normaliser = new Normalize();
    normaliser.setInputFormat(trainingInstances);
    trainingInstances = Filter.useFilter(trainingInstances, normaliser);

    // train
    algorithm.train(trainingInstances);
    // training summary
    trainingSummary = algorithm.getTrainingSummary(trainingInstances);
    totalTrainingInstances = trainingInstances.numInstances();
  }


  public double getDataReduction() {
    return algorithm.getDataReduction(totalTrainingInstances);
  }

  protected void performParameterValidation(Instances trainingInstances)
    throws Exception {
    // TODO
  }

  public double classifyInstance(Instance instance) throws Exception {
    if (algorithm == null) {
      throw new Exception("Algorithm has not been prepared.");
    }

    // normalise the instance
    normaliser.input(instance);
    normaliser.batchFinished();
    instance = normaliser.output();

    return algorithm.classify(instance);
  }

  public String toString() {
    StringBuilder buffer = new StringBuilder();
    buffer.append("Immunos-99 v1.0.\n");

    if (trainingSummary != null) {
      buffer.append("\n");
      buffer.append(trainingSummary);
    }

    return buffer.toString();
  }

  public String globalInfo() {
    StringBuilder buffer = new StringBuilder();
    buffer.append(toString());
    buffer.append("\n\n");

    buffer.append(" Jason Brownlee.  " +
      "[Technical Report].  " +
      "Immunos-81 - The Misunderstood Artificial Immune System.  " +
      "Victoria, Australia: " +
      "Centre for Intelligent Systems and Complex Processes (CISCP), " +
      "Faculty of Information and Communication Technologies (ICT), " +
      "Swinburne University of Technology; " +
      "2005 Feb; " +
      "Technical Report ID: 3-01. ");

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


  public void setOptions(String[] options) throws Exception {
    setTotalGenerations(OptionHelper.getInteger(PARAMETERS[0], options, 1));
    setSeed(OptionHelper.getLong(PARAMETERS[1], options, 1));
    setMinimumFitnessThreshold(OptionHelper.getDouble(PARAMETERS[2], options, -1));
    setSeedPopulationPercentage(OptionHelper.getDouble(PARAMETERS[3], options, 0.2));
    // parental option setting
    super.setOptions(options);
  }

  public String[] getOptions() {
    List<String> list = new ArrayList<String>(Arrays.asList(super.getOptions()));

    list.add("-" + PARAMETERS[0]);
    list.add(Integer.toString(totalGenerations));
    list.add("-" + PARAMETERS[1]);
    list.add(Long.toString(seed));
    list.add("-" + PARAMETERS[2]);
    list.add(Double.toString(minimumFitnessThreshold));
    list.add("-" + PARAMETERS[3]);
    list.add(Double.toString(seedPopulationPercentage));

    return list.toArray(new String[0]);
  }


  public String totalGenerationsTipText() {
    return DESCRIPTIONS[0];
  }

  public String seedTipText() {
    return DESCRIPTIONS[1];
  }

  public String minimumFitnessThresholdTipText() {
    return DESCRIPTIONS[2];
  }

  public String seedPopulationPercentageTipText() {
    return DESCRIPTIONS[3];
  }


  public double getMinimumFitnessThreshold() {
    return minimumFitnessThreshold;
  }

  public void setMinimumFitnessThreshold(double minimumFitnessThreshold) {
    this.minimumFitnessThreshold = minimumFitnessThreshold;
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

  public static void main(String[] args) {
   runClassifier(new Immunos99(), args);
  }

  /**
   * @return Returns the seedPopulationPercentage.
   */
  public double getSeedPopulationPercentage() {
    return seedPopulationPercentage;
  }

  /**
   * @param seedPopulationPercentage The seedPopulationPercentage to set.
   */
  public void setSeedPopulationPercentage(double seedPopulationPercentage) {
    this.seedPopulationPercentage = seedPopulationPercentage;
  }
}
