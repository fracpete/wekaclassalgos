package weka.classifiers.neural.singlelayerperceptron.algorithm;

import weka.classifiers.neural.common.CommonNeuralAlgorithmAncestor;
import weka.classifiers.neural.common.RandomWrapper;
import weka.classifiers.neural.common.SimpleNeuron;
import weka.classifiers.neural.common.initialisation.Initialisation;
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

public abstract class SLPAlgorithmAncestor extends CommonNeuralAlgorithmAncestor
{
    // neurons which make up this model
    protected final SimpleNeuron [] neurons;

    // learning rate function
    protected final LearningRateKernel learningRateFunction;
    
    
	public SLPAlgorithmAncestor(TransferFunction aTransfer,
							   double aBiasInput,
							   RandomWrapper aRand,
							   LearningRateKernel aKernel,
							   Instances trainingInstances)
	{
		super(aTransfer, aRand);
		
		learningRateFunction = aKernel;
		
		// determine the number of neurons required
		if(trainingInstances.classAttribute().isNumeric())
		{
			neurons = new SimpleNeuron[1];
		}
		// must be numeric
		else
		{
			neurons = new SimpleNeuron[trainingInstances.numClasses()];
		}

		// prepare the network structure
		prepareNetworkStructure(trainingInstances, aBiasInput);
	}
    
    
    


    public int getNumOutputNeurons()
    {
        if(neurons==null)
        {
            return 0;
        }

        return neurons.length;
    }
    
    
	public double [] getAllWeights()
	{
		if(neurons == null)
		{
			return null;
		}
		
		int totalWeights = neurons.length * neurons[0].getWeights().length;
		double [] weights = new double[totalWeights];
		int offset = 0;
		
		for (int i = 0; i < neurons.length; i++)
		{			
			double [] tmpWeights = neurons[i].getWeights();
			
			for (int k = 0; k < tmpWeights.length; k++, offset++)
			{
				weights[offset] = tmpWeights[k];
			}
		}
		
		return weights;
	}    




    protected abstract void calculateWeightErrors(Instance instance, SimpleNeuron neuron, double expected, double aLearningRate);


	public double getLearningRate(int aEpochNumber)
	{
		return learningRateFunction.currentLearningRate(aEpochNumber);
	}

    public void startingEpoch()
    {}

    public void finishedEpoch(Instances instances, double aLearningRate)
    {}



    public String getModelInformation()
    {
        StringBuffer buffer = new StringBuffer();

        buffer.append("Initial Learing Rate     : " + learningRateFunction.getInitialLearningRate() + "\n");
        buffer.append("Bias Input Value         : " + neurons[0].getBiasInputValue() + "\n");
        buffer.append("Output Layer Neurons     : " + neurons.length + "\n");

        return buffer.toString();
    }




    public void updateModel(Instance inputs, double aLearningRate)
    {
        // prepare an expected output vector
        double [] expected = prepareExpectedOutputVector(inputs);

        // calculate weight changes for each neuron
        for(int i=0; i<neurons.length; i++)
        {
            // calculate weight changes
            calculateWeightErrors(inputs, neurons[i], expected[i], aLearningRate);
        }
    }

    protected void prepareNetworkStructure(Instances instances, double aBiasInput)
    {
        // determine the number of attribtes
        int numAttributes = instances.numAttributes() - 1;

        // construct the required number of neurons
        for(int i=0; i<neurons.length; i++)
        {
            neurons[i] = new SimpleNeuron(numAttributes, aBiasInput);

            // initialise weights to between -0.5 and +0.5
            Initialisation.initialiseVectorToRandomWithSign(neurons[i].getWeights(), 0.5 , 0.0, rand);
        }
    }





    public double [] getNetworkOutputs(Instance instance)
    {
        double [] distribution = new double[neurons.length];

        for(int i=0; i<distribution.length; i++)
        {
            double activation = activate(neurons[i], instance);
            distribution[i] = transfer(activation);
        }

        return distribution;
    }

}