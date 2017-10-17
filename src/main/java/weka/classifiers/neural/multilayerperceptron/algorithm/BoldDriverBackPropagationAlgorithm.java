package weka.classifiers.neural.multilayerperceptron.algorithm;

import java.util.Enumeration;

import weka.classifiers.neural.common.RandomWrapper;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.common.transfer.TransferFunction;
import weka.core.Instance;
import weka.core.Instances;


/**
 * 
 * Date: 31/05/2004
 * File: BoldDriverBackPropagationAlgorithm.java
 * 
 * Adaptive learning rate algorithm, Bold Driver (Vogl’s Method)
 * 
 * @author Jason Brownlee
 *
 */
public class BoldDriverBackPropagationAlgorithm extends BackPropagationAlgorithm
{
    // the scale the error must increase before an increase is detected
    public final static double ERROR_INCREASE_SCALE = 1.05;


    // previous error value
    protected double previousErrorValue;

    // whether or not there was an increase in error last epoch
    protected boolean previousErrorWasIncrease;    
    
    protected final double errorDecreaseAdjustment;

    protected final double errorIncreaseAdjustment;
    
    protected double internalLearningRate = 0.0;
    
    
	public BoldDriverBackPropagationAlgorithm(TransferFunction aTransferFunction,
												RandomWrapper aRand,
												LearningRateKernel aLearningRateKernel,
												double aMomentum,
												double aWeightDecay,
												double aBiasValue,
												int [] aHiddenLayersTopology,
												Instances aTrainingInstances,
												double increase,
												double decrease)
	{
		super(aTransferFunction, aRand, aLearningRateKernel, aMomentum, aWeightDecay, aBiasValue, aHiddenLayersTopology, aTrainingInstances);
		errorIncreaseAdjustment = increase;
		errorDecreaseAdjustment = decrease;
		internalLearningRate = aLearningRateKernel.getInitialLearningRate();
	}



    public String getModelInformation()
    {
        StringBuffer buffer = new StringBuffer();

        buffer.append("Error Increase Adjustment: " + errorIncreaseAdjustment + "\n");
        buffer.append("Error Decrease Adjustment: " + errorDecreaseAdjustment + "\n");

        buffer.append(super.getModelInformation());

        return buffer.toString();
    }


    protected double calculateWeightDelta(double weightError,
                                          double lastDelta,
                                          double currentWeight,
                                          double aLearningRate)
    {
        // overriden
        // w(t+1) = w(t) + (lrate * error) + (momentum * lastWeightChange) - (weight decay * current weight)

        if(previousErrorWasIncrease)
        {
            // do not include momentum
            return (aLearningRate * weightError) - (weightDecay * currentWeight);
        }

        // include momentum
        return (aLearningRate * weightError) + (momentum * lastDelta) - (weightDecay * currentWeight);
    }



	public double getLearningRate(int epochNumber)
	{
		// controlled by the bold driver method
		return internalLearningRate;
	}


    public void finishedEpoch(Instances instances, double aLearningRate)
    {
        double sse = calculateSumSquaredErrors(instances);

        // check for decrease
        if(sse < previousErrorValue)
        {
            // learning rate is increased for error decreases
			internalLearningRate = (errorDecreaseAdjustment * aLearningRate);
            // not an increase
            previousErrorWasIncrease = false;
        }
        // check for increase
        else if(sse > (ERROR_INCREASE_SCALE * previousErrorValue))
        {
            // learning rate is decreased for error increases
			internalLearningRate = (errorIncreaseAdjustment * aLearningRate);
            // check if the last update was a decrease
            // so deltas are only cleared when they have to be
            if(!previousErrorWasIncrease)
            {
                // previous weight changes should not take effect for the next weight update (momentum)
                retractPreviousWeightDeltas();
            }

            // was an increase
            previousErrorWasIncrease = true;
        }
        // else no change

        // store error
        previousErrorValue = sse;
    }


    protected void retractPreviousWeightDeltas()
    {
        // iterate over all nodes and clear the previous weight
        // detlas used in momentum

        for(int i=0; i<neurons.length; i++)
        {
            for(int j=0; j<neurons[i].length; j++)
            {
                double [] deltas = neurons[i][j].getLastWeightDeltas();

                for(int k=0; k<deltas.length; k++)
                {
                    deltas[k] = 0.0;
                }
            }
        }
    }


    protected double calculateSumSquaredErrors(Instances instances)
    {
        Enumeration e = instances.enumerateInstances();
        double sse = 0.0;

        while(e.hasMoreElements())
        {
            Instance instance = (Instance) e.nextElement();

            // get model output: note outputs have already been calculated, simply reuse them
			double [] output = outputs[outputs.length-1];

            // determine expected output
            double [] expected = prepareExpectedOutputVector(instance);

            // calculate error for a single pattern
            double patternError = 0.0;

            for(int i=0; i<expected.length; i++)
            {
                // calculate error
                double error = (expected[i] - output[i]);

                // sum the square of the attribute errors
                patternError += (error * error);
            }

            // sum half of the pattern errors over all patterns
            sse += (0.5 * patternError);
        }

        return sse;
    }

}