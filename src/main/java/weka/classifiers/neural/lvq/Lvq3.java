package weka.classifiers.neural.lvq;

import weka.classifiers.Evaluation;
import weka.classifiers.neural.common.Constants;
import weka.classifiers.neural.common.learning.LearningKernelFactory;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.algorithm.Lvq3Algorithm;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Description: Implementation of the LVQ3 algorithm for use in WEKA
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class Lvq3 extends LvqAlgorithmAncestor {

  /**
   * Window size parameter
   */
  private final static String PARAM_WINDOW_SIZE = "W";

  /**
   * Epsilon parameter
   */
  private final static String PARAM_EPSILON = "E";

  /**
   * Window size parameter description
   */
  private final static String PARAM_WINDOW_SIZE_DESC = Constants.DESCRIPTION_WINDOW_SIZE;

  /**
   * Epsilon parameter description
   */
  private final static String PARAM_EPSILON_DESC = Constants.DESCRIPTION_EPSILON;

  /**
   * Window size value
   */
  protected double windowSize;

  /**
   * Epsilon value
   */
  protected double epsilon;


  public Lvq3() {
    // default values
    windowSize = 0.3;
    epsilon = 0.1;
  }


  protected void trainModel(Instances instances) {
    // construct the algorithm
    LearningRateKernel learningKernel = LearningKernelFactory.factory(learningFunction, learningRate, trainingIterations);
    Lvq3Algorithm algorithm = new Lvq3Algorithm(learningKernel, model, random, windowSize, epsilon);
    // add event listeners
    addEventListenersToAlgorithm(algorithm);
    // train the algorithm
    algorithm.trainModel(instances, trainingIterations);
  }


  /**
   * Responsible for validating algorithm specific parameters
   *
   * @throws Exception
   */
  protected void validateArguments() throws Exception {
    // window size can be anything

    // epsilon can be anything
  }

  /**
   * Returns a list of algorithm specific arguments
   *
   * @return Collection
   */
  protected Collection getListOptions() {
    ArrayList list = new ArrayList(2);

    list.add(new Option("\t" + PARAM_WINDOW_SIZE_DESC, PARAM_WINDOW_SIZE, 1, "-" + PARAM_WINDOW_SIZE + " <window size>"));
    list.add(new Option("\t" + PARAM_EPSILON_DESC, PARAM_EPSILON, 1, "-" + PARAM_EPSILON + " <epsilon learning rate>"));

    return list;
  }


  protected void setArguments(String[] options)
    throws Exception {
    // window
    String windowValue = Utils.getOption(PARAM_WINDOW_SIZE.charAt(0), options);
    if (hasValue(windowValue)) {
      windowSize = Double.parseDouble(windowValue);
    }

    // epsilon
    String epsilonValue = Utils.getOption(PARAM_EPSILON.charAt(0), options);
    if (hasValue(epsilonValue)) {
      epsilon = Double.parseDouble(epsilonValue);
    }
  }

  /**
   * Set an algorithm specific attribute
   *
   * @param name  - name of attribute
   * @param value - value of attribute
   * @return boolean - whether or not the attribute was set
   * @throws Exception
   */
  protected boolean setArgument(String name, String value) throws Exception {
    boolean found = false;

    if (("-" + PARAM_WINDOW_SIZE).equals(name)) {
      windowSize = Double.parseDouble(value);
      found = true;
    }
    else if (("-" + PARAM_EPSILON).equals(name)) {
      epsilon = Double.parseDouble(value);
      found = true;
    }

    return found;
  }

  /**
   * Returns a list of attributes and values
   *
   * @return Collection
   */
  protected Collection getAlgorithmOptions() {
    ArrayList list = new ArrayList(4);

    list.add("-" + PARAM_WINDOW_SIZE);
    list.add(Double.toString(windowSize));
    list.add("-" + PARAM_EPSILON);
    list.add(Double.toString(epsilon));

    return list;
  }

  /**
   * Returns information about this algorithm implementation
   *
   * @return String
   */
  public String globalInfo() {
    StringBuffer buffer = new StringBuffer(100);
    buffer.append("Learning Vector Quantisation (LVQ) - LVQ3.");
    buffer.append("The same as LVQ2.1, except only if the classes of the 2 BMUs match, ");
    buffer.append("otherwise, the a learning rate modified by the epsilon is used on both BMU's.");
    return buffer.toString();
  }

  /**
   * Window size tip
   *
   * @return
   */
  public String windowSizeTipText() {
    return PARAM_WINDOW_SIZE_DESC;
  }

  /**
   * Epsilon tip
   *
   * @return
   */
  public String epsilonTipText() {
    return PARAM_EPSILON_DESC;
  }

  /**
   * Entry point into the algorithm for direct usage
   *
   * @param args
   */
  public static void main(String[] args) {
    try {
      System.out.println(Evaluation.evaluateModel(new Lvq3(), args));
    }
    catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }

  /**
   * @return
   */
  public double getEpsilon() {
    return epsilon;
  }

  /**
   * @return
   */
  public double getWindowSize() {
    return windowSize;
  }

  /**
   * @param d
   */
  public void setEpsilon(double d) {
    epsilon = d;
  }

  /**
   * @param d
   */
  public void setWindowSize(double d) {
    windowSize = d;
  }

}