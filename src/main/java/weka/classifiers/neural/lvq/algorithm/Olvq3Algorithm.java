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

package weka.classifiers.neural.lvq.algorithm;

import weka.classifiers.neural.common.RandomWrapper;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.classifiers.neural.lvq.model.CommonModel;
import weka.classifiers.neural.lvq.model.LvqModel;
import weka.core.Instance;

/**
 * Date: 24/05/2004
 * File: OLVQ3.java
 *
 * @author Jason Brownlee
 */
public class Olvq3Algorithm extends Lvq3Algorithm {

  public Olvq3Algorithm(LearningRateKernel aLearningKernel,
			CommonModel aModel,
			RandomWrapper aRand,
			double aWindow,
			double aEpsilonRate) {
    super(aLearningKernel, aModel, aRand, aWindow, aEpsilonRate);
    // apply the learning rate to all codebook vectors
    model.applyLearningRateToAllVectors(learningKernel.getInitialLearningRate());
  }


  protected boolean usingGlobalLearningRate() {
    return false; // individual learning rates
  }


  protected void updateModel(Instance aInstance,
			     double aGlobalLearningRate) {
    // calculate distances to all codebook vectors
    CodebookVector[] bmus = ((LvqModel) model).get2Bmu(aInstance);

    // both bmu's must have different classes, one must have the correct
    // class and the distance ratio must be within the window
    if (bmusOfDifferentClassesAndInWindow(bmus[0], bmus[1], aInstance)) {
      // adjust the codebook vector using individual learning rate
      updateVector(bmus[0], aInstance, bmus[0].getIndividualLearningRate());
      updateVector(bmus[1], aInstance, bmus[1].getIndividualLearningRate());
      // adjust individual learning rates
      adjustIndividualLearningRate(bmus[0], aInstance);
      adjustIndividualLearningRate(bmus[1], aInstance);
    }
    // both bmu's are of the same class and match the expected class
    else if (bmusOfCorrectClass(bmus[0], bmus[1], aInstance)) {
      // adjust the codebook vector using individual learning rate * epsilonRate
      updateVector(bmus[0], aInstance, bmus[0].getIndividualLearningRate() * epsilonRate);
      updateVector(bmus[1], aInstance, bmus[1].getIndividualLearningRate() * epsilonRate);
      // adjust individual learning rates
      adjustIndividualLearningRate(bmus[0], aInstance);
      adjustIndividualLearningRate(bmus[1], aInstance);
    }
  }
}
