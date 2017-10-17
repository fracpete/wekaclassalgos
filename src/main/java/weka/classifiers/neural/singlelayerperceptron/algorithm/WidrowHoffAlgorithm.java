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

package weka.classifiers.neural.singlelayerperceptron.algorithm;

import weka.classifiers.neural.common.RandomWrapper;
import weka.classifiers.neural.common.SimpleNeuron;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.common.transfer.TransferFunction;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public class WidrowHoffAlgorithm extends SLPAlgorithmAncestor {

  public WidrowHoffAlgorithm(TransferFunction aTransfer,
			     double aBiasInput,
			     RandomWrapper aRand,
			     LearningRateKernel aKernel,
			     Instances trainingInstances) {
    super(aTransfer, aBiasInput, aRand, aKernel, trainingInstances);
  }


  protected void calculateWeightErrors(Instance instance,
				       SimpleNeuron neuron,
				       double expected,
				       double aLearningRate) {
    // Widrow-Hoff learning rule: delta = LearningRate * (Target - Activation) * Input

    int offset = 0;

    // calculate the activation for the neuron
    double activation = activate(neuron, instance);

    // get the node weights
    double[] weights = neuron.getWeights();

    // udpate neuron weights
    for (int i = 0; i < instance.numAttributes(); i++) {
      // class is not an attribute
      if (i != instance.classIndex()) {
	// never adjust the weight connected to a missing value
	// it is not included in thew activation, thus has no impact in the result
	if (instance.isMissing(i)) {
	  offset++;
	}
	else {
	  // perceptron learning rule:
	  // delta = LearningRate * (Target - Activation) * Input
	  weights[offset++] += (aLearningRate * (expected - activation) * instance.value(i));
	}
      }
    }

    // update the weight on this bias
    offset = neuron.getBiasIndex();
    weights[offset] += (aLearningRate * (expected - activation) * neuron.getBiasInputValue());
  }

}