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

package weka.classifiers.neural.common.training;

import weka.classifiers.neural.common.BatchTrainableNeuralModel;
import weka.classifiers.neural.common.NeuralModel;
import weka.classifiers.neural.common.RandomWrapper;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public class BatchTrainer extends NeuralTrainer {

  public BatchTrainer(RandomWrapper aRand) {
    super(aRand);
  }

  public void trainModel(NeuralModel aModel,
			 Instances aInstances,
			 int numIterations) {
    BatchTrainableNeuralModel model = (BatchTrainableNeuralModel) aModel;
    Instances epochInstances = new Instances(aInstances);

    // train until we can stop
    for (int iteration = 0; iteration < numIterations; iteration++) {
      // prepare the model for an epoch
      aModel.startingEpoch();

      // get the learning rate
      double learingRate = aModel.getLearningRate(iteration);

      // randomize the dataset
      epochInstances.randomize(rand.getRand());

      // perform a single epoch
      Enumeration e = epochInstances.enumerateInstances();
      while (e.hasMoreElements()) {
	// get an instance
	Instance instance = (Instance) e.nextElement();

	// calculate weight changes
	model.calculateWeightErrors(instance, learingRate);
      }

      // apply and clear weight changes at the end of the epoch
      model.applyWeightDeltas(learingRate);

      // finished epoch
      aModel.finishedEpoch(epochInstances, learingRate);
    }
  }

}