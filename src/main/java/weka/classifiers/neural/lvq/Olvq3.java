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
import weka.classifiers.neural.lvq.algorithm.Olvq3Algorithm;
import weka.core.Instances;

/**
 * Date: 24/05/2004
 * File: OLVQ3.java
 *
 * @author Jason Brownlee
 */
public class Olvq3 extends Lvq3 {

  protected void trainModel(Instances instances) {
    // construct the algorithm
    LearningRateKernel learningKernel = LearningKernelFactory.factory(learningFunction, learningRate, trainingIterations);
    Olvq3Algorithm algorithm = new Olvq3Algorithm(learningKernel, model, random, windowSize, epsilon);
    // add event listeners
    addEventListenersToAlgorithm(algorithm);
    // train the algorithm
    algorithm.trainModel(instances, trainingIterations);
  }


  /**
   * Returns information about this algorithm implementation
   *
   * @return String
   */
  public String globalInfo() {
    StringBuffer buffer = new StringBuffer(100);
    buffer.append("Learning Vector Quantisation (LVQ) - OLVQ1.");
    buffer.append("The same as the LVQ3 algorithm, except each codebook vector has its ");
    buffer.append("own individual learning rate (rather than a global learning rate) in the same manner as OLVQ1.");
    return buffer.toString();
  }

  /**
   * Entry point into the algorithm for direct usage
   *
   * @param args
   */
  public static void main(String[] args) {
   runClassifier(new Olvq3(), args);
  }
}
