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

package weka.classifiers.neural.common;

import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public interface NeuralModel extends Serializable {

  // update the model for a single pattern
  public void updateModel(Instance instance, double aLearningRate);

  public void startingEpoch();

  public void finishedEpoch(Instances instances, double aLearningRate);

  public double getLearningRate(int epochNumber);


  // run a vector through the model and get a result
  public double[] getNetworkOutputs(Instance instance);

  public double[] getDistributionForInstance(Instance instance);

  // stats
  public String getModelInformation();

  public int getNumOutputNeurons();

  public double[] getAllWeights();
}