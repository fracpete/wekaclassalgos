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

package weka.classifiers.neural.lvq;

import weka.classifiers.Evaluation;
import weka.classifiers.neural.common.Constants;
import weka.classifiers.neural.common.learning.LearningKernelFactory;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.algorithm.Lvq2_1Algorithm;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Description: Implementation of the LVQ2.1 algorithm for use in WEKA.
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class Lvq2_1 extends LvqAlgorithmAncestor {

  /**
   * Window size argument
   */
  private final static String PARAM_WINDOW_SIZE = "W"; // window size

  /**
   * Window size argument description
   */
  private final static String PARAM_WINDOW_SIZE_DESC = Constants.DESCRIPTION_WINDOW_SIZE;

  /**
   * Window size value
   */
  protected double windowSize;


  public Lvq2_1() {
    // set default values
    windowSize = 0.3;
  }


  protected void trainModel(Instances instances) {
    // construct the algorithm
    LearningRateKernel learningKernel = LearningKernelFactory.factory(learningFunction, learningRate, trainingIterations);
    Lvq2_1Algorithm algorithm = new Lvq2_1Algorithm(learningKernel, model, random, windowSize);
    // add event listeners
    addEventListenersToAlgorithm(algorithm);
    // train the algorithm
    algorithm.trainModel(instances, trainingIterations);
  }


  /**
   * Validate algorithm specific arguments
   *
   * @throws Exception
   */
  protected void validateArguments() throws Exception {
    // window size can be anything
  }

  /**
   * Returns a list of algorithm specific options
   *
   * @return Collection
   */
  protected Collection getListOptions() {
    ArrayList list = new ArrayList(1);
    list.add(new Option("\t" + PARAM_WINDOW_SIZE_DESC, PARAM_WINDOW_SIZE, 1, "-" + PARAM_WINDOW_SIZE + " <window sizes>"));
    return list;
  }


  protected void setArguments(String[] options)
    throws Exception {
    String windowValue = Utils.getOption(PARAM_WINDOW_SIZE.charAt(0), options);

    if (hasValue(windowValue)) {
      windowSize = Double.parseDouble(windowValue);
    }
  }

  /**
   * Returns a list of algorithm options and values
   *
   * @return Collection
   */
  protected Collection getAlgorithmOptions() {
    ArrayList list = new ArrayList(2);
    list.add("-" + PARAM_WINDOW_SIZE);
    list.add(Double.toString(windowSize));
    return list;
  }

  /**
   * Returns global info on the algorithm implementation
   *
   * @return String
   */
  public String globalInfo() {
    StringBuffer buffer = new StringBuffer(100);
    buffer.append("Learning Vector Quantisation (LVQ) - LVQ2_1.");
    buffer.append("The top two BMU's (best matching units) are selected for a data vector. ");
    buffer.append("One of the BMU's class's must match the data vector, and the vectors ");
    buffer.append("are within the window: min (distanceI/distanceJ, distanceJ/distanceI) > s, where s = (1-window)/(1+window).");
    return buffer.toString();
  }

  /**
   * Window size tool tip
   *
   * @return
   */
  public String windowSizeTipText() {
    return PARAM_WINDOW_SIZE_DESC;
  }

  /**
   * Set the window size value
   *
   * @param w
   */
  public void setWindowSize(double w) {
    windowSize = w;
  }

  /**
   * Returns the window size value
   *
   * @return
   */
  public double getWindowSize() {
    return windowSize;
  }

  /**
   * Entry point into the algorithm for direct usage
   *
   * @param args
   */
  public static void main(String[] args) {
    try {
      System.out.println(Evaluation.evaluateModel(new Lvq2_1(), args));
    }
    catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }
}