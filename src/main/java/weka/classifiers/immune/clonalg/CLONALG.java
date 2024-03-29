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
public class CLONALG extends AbstractClassifier {

  protected double clonalFactor; // beta

  protected int antibodyPoolSize; // N

  protected int selectionPoolSize; // n

  protected int totalReplacement; // d

  protected int numGenerations; // Ngen

  protected long seed; // random number seed

  protected double remainderPoolRatio; // typically 5%-8%

  protected CLONALGAlgorithm algorithm;

  private final static String[] PARAMETERS =
    {
      "B",
      "N",
      "n",
      "D",
      "G",
      "S",
      "R"
    };

  private final static String[] DESCRIPTIONS =
    {
      "Clonal factor (beta). Used to scale the number of clones created by the selected best antibodies.",
      "Antibody pool size (N). The total antibodies maintained in the memory pool and remainder pool.",
      "Selection pool size (n). The total number of best antibodies selected for cloning and mutation each iteration.",
      "Total replacements (d). The total number of antibodies in the remainder pool that are replaced each iteration. Typically 5%-8%",
      "Total generations. The total number of times that all antigens are exposed to the system.",
      "Random number generator seed. Seed used to initialise the random number generator.",
      "Remainder pool percentage. The percentage of the total antibody pool size allocated for the remainder pool."
    };


  public CLONALG() {
    // set defaults
    clonalFactor = 0.1; // beta
    antibodyPoolSize = 30; // N
    selectionPoolSize = 20; // n
    totalReplacement = 0; // d
    numGenerations = 10;
    seed = 1;
    remainderPoolRatio = 0.1;
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

    // construct trainer
    algorithm = new CLONALGAlgorithm(
      clonalFactor,
      antibodyPoolSize,
      selectionPoolSize,
      totalReplacement,
      numGenerations,
      seed,
      remainderPoolRatio);

    // train
    algorithm.train(trainingInstances);
  }

  public double classifyInstance(Instance instance) throws Exception {
    if (algorithm == null) {
      throw new Exception("Algorithm has not been prepared.");
    }
    return algorithm.classify(instance);
  }

  public String toString() {
    StringBuilder buffer = new StringBuilder();
    buffer.append("CLONALG v1.0.\n");
    return buffer.toString();
  }

  public String globalInfo() {
    StringBuilder buffer = new StringBuilder();
    buffer.append(toString());
    buffer.append("CLONALG - Clonal Selection Algorithm for classifiation.");
    buffer.append("\n\n");

    buffer.append(
      "Leandro N. de Castro and Fernando J. Von Zuben. " +
	"Learning and Optimization Using the Clonal Selection Principle. " +
	"IEEE Transactions on Evolutionary Computation, Special Issue on Artificial Immune Systems. " +
	"2002; " +
	"6(3): " +
	"239-251."
    );

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
    setClonalFactor(OptionHelper.getDouble(PARAMETERS[0], options, 0.1));
    setAntibodyPoolSize(OptionHelper.getInteger(PARAMETERS[1], options, 30));
    setSelectionPoolSize(OptionHelper.getInteger(PARAMETERS[2], options, 20));
    setTotalReplacement(OptionHelper.getInteger(PARAMETERS[3], options, 0));
    setNumGenerations(OptionHelper.getInteger(PARAMETERS[4], options, 10));
    setSeed(OptionHelper.getLong(PARAMETERS[5], options, 1));
    setRemainderPoolRatio(OptionHelper.getDouble(PARAMETERS[6], options, 0.1));
    // parental option setting
    super.setOptions(options);
  }

  public String[] getOptions() {
    List<String> list = new ArrayList<String>(Arrays.asList(super.getOptions()));

    list.add("-" + PARAMETERS[0]);
    list.add(Double.toString(clonalFactor));
    list.add("-" + PARAMETERS[1]);
    list.add(Integer.toString(antibodyPoolSize));
    list.add("-" + PARAMETERS[2]);
    list.add(Integer.toString(selectionPoolSize));
    list.add("-" + PARAMETERS[3]);
    list.add(Integer.toString(totalReplacement));
    list.add("-" + PARAMETERS[4]);
    list.add(Integer.toString(numGenerations));
    list.add("-" + PARAMETERS[5]);
    list.add(Long.toString(seed));
    list.add("-" + PARAMETERS[6]);
    list.add(Double.toString(remainderPoolRatio));

    return list.toArray(new String[0]);
  }


  public String clonalFactorTipText() {
    return DESCRIPTIONS[0];
  }

  public String antibodyPoolSizeTipText() {
    return DESCRIPTIONS[1];
  }

  public String selectionPoolSizeTipText() {
    return DESCRIPTIONS[2];
  }

  public String totalReplacementTipText() {
    return DESCRIPTIONS[3];
  }

  public String numGenerationsTipText() {
    return DESCRIPTIONS[4];
  }

  public String seedTipText() {
    return DESCRIPTIONS[5];
  }

  public String remainderPoolRatioTipText() {
    return DESCRIPTIONS[6];
  }


  /**
   * @return Returns the antibodyPoolSize.
   */
  public int getAntibodyPoolSize() {
    return antibodyPoolSize;
  }

  /**
   * @param antibodyPoolSize The antibodyPoolSize to set.
   */
  public void setAntibodyPoolSize(int antibodyPoolSize) {
    this.antibodyPoolSize = antibodyPoolSize;
  }

  /**
   * @return Returns the clonalFactor.
   */
  public double getClonalFactor() {
    return clonalFactor;
  }

  /**
   * @param clonalFactor The clonalFactor to set.
   */
  public void setClonalFactor(double clonalFactor) {
    this.clonalFactor = clonalFactor;
  }

  /**
   * @return Returns the numGenerations.
   */
  public int getNumGenerations() {
    return numGenerations;
  }

  /**
   * @param numGenerations The numGenerations to set.
   */
  public void setNumGenerations(int numGenerations) {
    this.numGenerations = numGenerations;
  }

  /**
   * @return Returns the remainderPoolRatio.
   */
  public double getRemainderPoolRatio() {
    return remainderPoolRatio;
  }

  /**
   * @param repertoirePoolRatio The repertoirePoolRatio to set.
   */
  public void setRemainderPoolRatio(double remainder) {
    this.remainderPoolRatio = remainder;
  }

  /**
   * @return Returns the seed.
   */
  public long getSeed() {
    return seed;
  }

  /**
   * @param seed The seed to set.
   */
  public void setSeed(long seed) {
    this.seed = seed;
  }

  /**
   * @return Returns the selectionPoolSize.
   */
  public int getSelectionPoolSize() {
    return selectionPoolSize;
  }

  /**
   * @param selectionPoolSize The selectionPoolSize to set.
   */
  public void setSelectionPoolSize(int selectionPoolSize) {
    this.selectionPoolSize = selectionPoolSize;
  }

  /**
   * @return Returns the totalReplacement.
   */
  public int getTotalReplacement() {
    return totalReplacement;
  }

  /**
   * @param totalReplacement The totalReplacement to set.
   */
  public void setTotalReplacement(int totalReplacement) {
    this.totalReplacement = totalReplacement;
  }

  public static void main(String[] args) {
   runClassifier(new CLONALG(), args);
  }
}
