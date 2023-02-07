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

import weka.classifiers.neural.common.learning.LearningKernelFactory;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.algorithm.Lvq1Algorithm;
import weka.core.Instances;
import weka.core.Option;

import java.util.Collection;


/**
 * Description: Implementation of the LVQ1 algorithm for use in WEKA
 * Implements elements required for the common LVQ algorithm framework
 * specific to the LVQ1 algorithm.
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class Lvq1 extends LvqAlgorithmAncestor {


  protected void trainModel(Instances instances) {
    // construct the algorithm
    LearningRateKernel learningKernel = LearningKernelFactory.factory(learningFunction, learningRate, trainingIterations);
    Lvq1Algorithm algorithm = new Lvq1Algorithm(learningKernel, model, random);
    // add event listeners
    addEventListenersToAlgorithm(algorithm);
    // train the algorithm
    algorithm.trainModel(instances, trainingIterations);
  }

  /**
   * Validate LVQ1 specific arguments
   *
   * @throws Exception
   */
  protected void validateArguments()
    throws Exception {
    // do nothing
  }

  /**
   * Provide list of LVQ1 specific options
   *
   * @return Collection
   */
  protected Collection<Option> getListOptions() {
    // do nothing
    return null;
  }

  protected void setArguments(String[] options)
    throws Exception {
  }

  /**
   * Provide collection of LVQ1 specific options
   *
   * @return Collection
   */
  protected Collection<String> getAlgorithmOptions() {
    // do nothing
    return null;
  }

  /**
   * Return LVQ1 specific information
   *
   * @return String
   */
  public String globalInfo() {
    StringBuilder buffer = new StringBuilder();
    buffer.append("Learning Vector Quantisation (LVQ) - LVQ1.");
    buffer.append("A single BMU (best matching unit) is selected and moved closer or ");
    buffer.append("further away from each data vector, per iteration.");
    return buffer.toString();
  }

  /**
   * Entry point into the algorithm for direct usage
   *
   * @param args
   */
  public static void main(String[] args) {
    runClassifier(new Lvq1(), args);
  }
}