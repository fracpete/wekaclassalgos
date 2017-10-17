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
import weka.classifiers.neural.lvq.model.ModelUpdater;
import weka.classifiers.neural.lvq.model.SomModel;
import weka.classifiers.neural.lvq.neighborhood.NeighbourhoodKernel;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Date: 25/05/2004
 * File: SOMAlgorithm.java
 *
 * @author Jason Brownlee
 */
public class SomAlgorithm extends CommonAncestor {

  protected final NeighbourhoodKernel neighbourhoodKernel;


  public SomAlgorithm(LearningRateKernel aLearningKernel,
		      NeighbourhoodKernel aNeighborhoodKernel,
		      SomModel aModel,
		      RandomWrapper aRand,
		      boolean isSupervised) {
    super(aLearningKernel, aModel, aRand, isSupervised);
    neighbourhoodKernel = aNeighborhoodKernel;
  }

  public void labelMap(Instances aInstances) {
    for (int i = 0; i < aInstances.numInstances(); i++) {
      // get the bmu
      CodebookVector bmu = model.getBmu(aInstances.instance(i));
      // set the class for the bmu
      bmu.setClassification(aInstances.instance(i).classValue());
    }
  }


  public void trainModel(Instances aInstances, int numIterations) {
    SomModelUpdator updator = new SomModelUpdator();

    for (int i = 0; i < numIterations; i++) {
      // select an instance
      Instance selectedInstance = selectRandomInstance(aInstances);
      // update the model
      updateModel(selectedInstance, i, updator);
      // send events
      activateEpochEventListeners(i, numIterations);
    }
  }


  protected void updateModel(Instance aInstance, int iteration, SomModelUpdator updator) {
    // find the bmu
    CodebookVector bmu = model.getBmu(aInstance);
    // determine current neighbourood size
    double neighbourhoodSize = neighbourhoodKernel.currentNeighborhoodSize(iteration);
    // determine current learning rate
    double learningRate = learningKernel.currentLearningRate(iteration);
    // store values in the model updator
    ((SomModelUpdator) updator).prepareToUpdateModel(bmu, aInstance, neighbourhoodSize, learningRate);
    // update the model
    model.updateModel(updator);
  }


  protected class SomModelUpdator implements ModelUpdater {

    private CodebookVector currentBmu;

    private double currentNeighbourhoodSize;

    private double currentLearningRate;

    private Instance currentInstance;

    public void prepareToUpdateModel(CodebookVector aBmu,
				     Instance aInstance,
				     double aNeighbourhoodSize,
				     double aCurrentLearningRate) {
      currentBmu = aBmu;
      currentNeighbourhoodSize = aNeighbourhoodSize;
      currentLearningRate = aCurrentLearningRate;
      currentInstance = aInstance;
    }

    public void updateCodebookVector(CodebookVector aCodebookVector) {
      // calculate distance from the model
      double distance = ((SomModel) model).calculateNeighbourhoodDistance(currentBmu, aCodebookVector);
      // check if distance is within the neighbourhood
      if (neighbourhoodKernel.isDistanceInRadius(distance, currentNeighbourhoodSize)) {
	// calculate adjusted learning rate
	double learningRate = neighbourhoodKernel.calculateNeighbourhoodAdjustedLearningRate(currentLearningRate, distance, currentNeighbourhoodSize);
	// update the attributes in the codebook vector
	updateVector(aCodebookVector, currentInstance, learningRate);
      }
    }
  }
}
