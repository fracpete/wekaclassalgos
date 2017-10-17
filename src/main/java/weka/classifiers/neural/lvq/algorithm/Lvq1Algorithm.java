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

/**
 * Description: Implementation of the LVQ algorithm used to construct a model
 * for a given dataset
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class Lvq1Algorithm extends LVQAlgorithmAncestor {

  public Lvq1Algorithm(LearningRateKernel aLearningKernel,
		       CommonModel aModel,
		       RandomWrapper aRand) {
    super(aLearningKernel, aModel, aRand);
  }


  protected boolean usingGlobalLearningRate() {
    return true;
  }

  protected void updateModel(Instance aInstance,
			     double currentLearningRate) {
    // reference the bmu
    CodebookVector bmu = model.getBmu(aInstance);
    // adjust the codebook vector
    updateVector(bmu, aInstance, currentLearningRate);
  }
}