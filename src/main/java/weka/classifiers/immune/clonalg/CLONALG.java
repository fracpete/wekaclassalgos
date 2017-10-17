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
    StringBuffer buffer = new StringBuffer(1000);
    buffer.append("CLONALG v1.0.\n");
    return buffer.toString();
  }

  public String globalInfo() {
    StringBuffer buffer = new StringBuffer(1000);
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
    // parental option setting
    super.setOptions(options);

    setClonalFactor(getDouble(PARAMETERS[0], options));
    setAntibodyPoolSize(getInteger(PARAMETERS[1], options));
    setSelectionPoolSize(getInteger(PARAMETERS[2], options));
    setTotalReplacement(getInteger(PARAMETERS[3], options));
    setNumGenerations(getInteger(PARAMETERS[4], options));
    setSeed(getLong(PARAMETERS[5], options));
    setRemainderPoolRatio(getDouble(PARAMETERS[6], options));
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


  public String[] getOptions() {
    LinkedList<String> list = new LinkedList<String>();

    String[] options = super.getOptions();
    for (int i = 0; i < options.length; i++) {
      list.add(options[i]);
    }

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

    return list.toArray(new String[list.size()]);
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

  public static void main(String[] argv) {

    try {
      System.out.println(Evaluation.evaluateModel(new CLONALG(), argv));
    }
    catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }
}
