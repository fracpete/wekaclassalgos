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
 * @author Jason Brownlee
 * @version 1.0
 */

public class WidrowHoffAlgorithm extends SLPAlgorithmAncestor
{

	public WidrowHoffAlgorithm(TransferFunction aTransfer,
							   double aBiasInput,
							   RandomWrapper aRand,
							   LearningRateKernel aKernel,
							   Instances trainingInstances)
	{
		super(aTransfer, aBiasInput, aRand, aKernel, trainingInstances);
	}


    protected void calculateWeightErrors(Instance instance,
                                         SimpleNeuron neuron,
                                         double expected,
										 double aLearningRate)
    {
        // Widrow-Hoff learning rule: delta = LearningRate * (Target - Activation) * Input

        int offset = 0;

        // calculate the activation for the neuron
        double activation = activate(neuron, instance);

        // get the node weights
        double [] weights = neuron.getWeights();

        // udpate neuron weights
        for(int i=0; i<instance.numAttributes(); i++)
        {
            // class is not an attribute
            if(i != instance.classIndex())
            {
                // never adjust the weight connected to a missing value
                // it is not included in thew activation, thus has no impact in the result
                if(instance.isMissing(i))
                {
                    offset++;
                }
                else
                {
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