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
import weka.core.Instance;
import weka.core.Instances;


/**
 * Description: Common ancestor for LVQ algorithm implementations,
 * specifically used for building models from training datasets
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public abstract class LVQAlgorithmAncestor extends CommonAncestor {

  public LVQAlgorithmAncestor(LearningRateKernel aLearningKernel,
			      CommonModel aModel,
			      RandomWrapper aRand) {
    super(aLearningKernel, aModel, aRand, true);
  }


  protected abstract void updateModel(Instance aInstances, double currentLearningRate);

  protected abstract boolean usingGlobalLearningRate();


  public void trainModel(Instances aInstances, int numIterations) {
    // initialise to the inital learning rate
    double currentLearningRate = learningKernel.getInitialLearningRate();

    for (int i = 0; i < numIterations; i++) {
      // attempt to avoid an unncessary calculation
      if (usingGlobalLearningRate()) {
	// learning rate for this iteration
	currentLearningRate = learningKernel.currentLearningRate(i);
      }
      // select a random data instance
      Instance selectedInstance = selectRandomInstance(aInstances);
      // update the model using LVQ algorithm
      updateModel(selectedInstance, currentLearningRate);
      // send events
      activateEpochEventListeners(i, numIterations);
    }
  }


  /**
   * Used for adjusting individual bmu learning rates consistantly
   *
   * @param aBmu
   * @param aInstance
   */
  protected void adjustIndividualLearningRate(CodebookVector aBmu, Instance aInstance) {
    double rate = aBmu.getIndividualLearningRate();

    if (isSameClass(aInstance, aBmu)) {
      // decrease the rate because it was correct
      aBmu.setIndividualLearningRate(rate / (1.0 + rate));
    }
    else {
      // increase the rate because it was incorrect
      aBmu.setIndividualLearningRate(rate / (1.0 - rate));
      // check for getting two large
      if (aBmu.getIndividualLearningRate() > learningKernel.getInitialLearningRate()) {
	aBmu.setIndividualLearningRate(learningKernel.getInitialLearningRate());
      }
    }
  }
}