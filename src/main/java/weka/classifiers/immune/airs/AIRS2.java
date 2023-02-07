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
package weka.classifiers.immune.airs;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.immune.airs.algorithm.AIRS2Trainer;
import weka.classifiers.immune.airs.algorithm.AISModelClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHelper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Vector;

/**
 * Type: AIRS1
 * File: AIRS1.java
 * Date: 07/01/2005
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class AIRS2 extends AbstractClassifier
  implements AIRSParameterDocumentation {

  // paramters
  protected long seed;

  protected double affinityThresholdScalar;

  protected double clonalRate;

  protected double hypermutationRate;

  protected double totalResources;

  protected double stimulationValue;

  protected int numInstancesAffinityThreshold;

  protected int memInitialPoolSize;

  protected int knn;


  protected String trainingSummary;

  protected String classifierSummary;


  private final static String[] PARAMETERS =
    {
      "S", // seed
      "F", // affinity threshold
      "C", // clonal rate
      "H", // hypermutation
      "R", // total resources
      "V", // stimulation value
      "A", // num affinity threshold instances
      "E", // mem pool size
      "K"  // kNN
    };


  private final static String[] DESCRIPTIONS =
    {
      PARAM_SEED,
      PARAM_ATS,
      PARAM_CLONAL_RATE,
      PARAM_HMR,
      PARAM_RESOURCES,
      PARAM_STIMULATION,
      PARAM_AT_INSTANCES,
      PARAM_MEM_INSTANCES,
      PARAM_KNN
    };


  /**
   * The model
   */
  protected AISModelClassifier classifier;


  public AIRS2() {
    // set default values
    seed = 1;
    affinityThresholdScalar = 0.2;
    totalResources = 150;
    stimulationValue = 0.9;
    clonalRate = 10;
    hypermutationRate = 2.0;
    numInstancesAffinityThreshold = -1;
    memInitialPoolSize = 1;
    knn = 3;
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

  /**
   * @param data
   * @throws java.lang.Exception
   */
  public void buildClassifier(Instances data) throws Exception {
    Instances trainingInstances = new Instances(data);
    trainingInstances.deleteWithMissingClass();

    getCapabilities().testWithFail(trainingInstances);

    // validate paramters
    validateParameters(trainingInstances);

    // construct trainer
    Random rand = new Random(seed);

    AIRS2Trainer trainer = new AIRS2Trainer(
      affinityThresholdScalar,
      clonalRate,
      hypermutationRate,
      totalResources,
      stimulationValue,
      numInstancesAffinityThreshold,
      rand,
      memInitialPoolSize,
      knn);

    // prepare classifier
    classifier = trainer.train(trainingInstances);

    // get summaries
    trainingSummary = trainer.getTrainingSummary();
    classifierSummary = classifier.getModelSummary(trainingInstances);
  }


  protected void validateParameters(Instances trainingInstances)
    throws Exception {
    int numInstances = trainingInstances.numInstances();

    if (memInitialPoolSize > numInstances) {
      memInitialPoolSize = numInstances;
    }
  }


  public double classifyInstance(Instance instance)
    throws Exception {
    if (classifier == null) {
      throw new Exception("Algorithm has not been prepared.");
    }

    // TODO: validate of data provided matches training data specs

    return classifier.classifyInstance(instance);
  }


  public String toString() {
    StringBuilder buffer = new StringBuilder();
    buffer.append("AIRS2 - Artificial Immune Recognition System v2.0\n");
    buffer.append("\n");

    if (trainingSummary != null) {
      buffer.append(trainingSummary);
      buffer.append("\n");
    }

    if (classifierSummary != null) {
      buffer.append(classifierSummary);
    }

    return buffer.toString();
  }


  public String globalInfo() {
    StringBuilder buffer = new StringBuilder();
    buffer.append(toString());
    buffer.append("A resource limited artifical immune system (AIS) ");
    buffer.append("for supervised classification, using clonal selection, ");
    buffer.append("affinity maturation and affinity recognition balls (ARBs).");
    buffer.append("\n\n");

    buffer.append("Andrew Watkins, Jon Timmis, and Lois Boggess, ");
    buffer.append("Artificial Immune Recognition System (AIRS): An Immune-Inspired Supervised Learning Algorithm, ");
    buffer.append("Genetic Programming and Evolvable Machines, ");
    buffer.append("vol. 5, pp. 291-317, Sep, 2004.");

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
    // long
    setSeed(OptionHelper.getLong(PARAMETERS[0], options, 1));
    // double
    setAffinityThresholdScalar(OptionHelper.getDouble(PARAMETERS[1], options, 0.2));
    setClonalRate(OptionHelper.getDouble(PARAMETERS[2], options, 10));
    setHypermutationRate(OptionHelper.getDouble(PARAMETERS[3], options, 2.0));
    setTotalResources(OptionHelper.getDouble(PARAMETERS[4], options, 150));
    setStimulationValue(OptionHelper.getDouble(PARAMETERS[5], options, 0.9));
    // int
    setNumInstancesAffinityThreshold(OptionHelper.getInteger(PARAMETERS[6], options, -1));
    setMemInitialPoolSize(OptionHelper.getInteger(PARAMETERS[7], options, 1));
    setKnn(OptionHelper.getInteger(PARAMETERS[8], options, 3));
    // parental option setting
    super.setOptions(options);
  }


  public String[] getOptions() {
    List<String> list = new ArrayList<String>(Arrays.asList(super.getOptions()));

    // long
    list.add("-" + PARAMETERS[0]);
    list.add(Long.toString(seed));
    // double
    list.add("-" + PARAMETERS[1]);
    list.add(Double.toString(affinityThresholdScalar));
    list.add("-" + PARAMETERS[2]);
    list.add(Double.toString(clonalRate));
    list.add("-" + PARAMETERS[3]);
    list.add(Double.toString(hypermutationRate));
    list.add("-" + PARAMETERS[4]);
    list.add(Double.toString(totalResources));
    list.add("-" + PARAMETERS[5]);
    list.add(Double.toString(stimulationValue));
    // int
    list.add("-" + PARAMETERS[6]);
    list.add(Integer.toString(numInstancesAffinityThreshold));
    list.add("-" + PARAMETERS[7]);
    list.add(Integer.toString(memInitialPoolSize));
    list.add("-" + PARAMETERS[8]);
    list.add(Integer.toString(knn));

    return list.toArray(new String[0]);
  }


  // long
  public String seedTipText() {
    return DESCRIPTIONS[0];
  }

  // double
  public String affinityThresholdScalarTipText() {
    return DESCRIPTIONS[1];
  }

  public String clonalRateTipText() {
    return DESCRIPTIONS[2];
  }

  public String hypermutationRateTipText() {
    return DESCRIPTIONS[3];
  }

  public String totalResourcesTipText() {
    return DESCRIPTIONS[4];
  }

  public String stimulationValueTipText() {
    return DESCRIPTIONS[5];
  }

  // int
  public String numInstancesAffinityThresholdTipText() {
    return DESCRIPTIONS[6];
  }

  public String memInitialPoolSizeTipText() {
    return DESCRIPTIONS[7];
  }

  public String knnTipText() {
    return DESCRIPTIONS[8];
  }


  public double getAffinityThresholdScalar() {
    return affinityThresholdScalar;
  }

  public void setAffinityThresholdScalar(double affinityThresholdScalar) {
    this.affinityThresholdScalar = affinityThresholdScalar;
  }

  public AISModelClassifier getClassifier() {
    return classifier;
  }

  public void setClassifier(AISModelClassifier classifier) {
    this.classifier = classifier;
  }

  public double getClonalRate() {
    return clonalRate;
  }

  public void setClonalRate(double clonalRate) {
    this.clonalRate = clonalRate;
  }

  public double getHypermutationRate() {
    return hypermutationRate;
  }

  public void setHypermutationRate(double hypermutationRate) {
    this.hypermutationRate = hypermutationRate;
  }

  public int getKnn() {
    return knn;
  }

  public void setKnn(int knn) {
    this.knn = knn;
  }

  public int getMemInitialPoolSize() {
    return memInitialPoolSize;
  }

  public void setMemInitialPoolSize(int memInitialPoolSize) {
    this.memInitialPoolSize = memInitialPoolSize;
  }

  public int getNumInstancesAffinityThreshold() {
    return numInstancesAffinityThreshold;
  }

  public void setNumInstancesAffinityThreshold(int numInstancesAffinityThreshold) {
    this.numInstancesAffinityThreshold = numInstancesAffinityThreshold;
  }

  public long getSeed() {
    return seed;
  }

  public void setSeed(long seed) {
    this.seed = seed;
  }

  public double getStimulationValue() {
    return stimulationValue;
  }

  public void setStimulationValue(double stimulationValue) {
    this.stimulationValue = stimulationValue;
  }

  public double getTotalResources() {
    return totalResources;
  }

  public void setTotalResources(double totalResources) {
    this.totalResources = totalResources;
  }


  public static void main(String[] args) {
   runClassifier(new AIRS2(), args);
  }
}
