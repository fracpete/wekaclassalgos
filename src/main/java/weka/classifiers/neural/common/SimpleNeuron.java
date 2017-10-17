package weka.classifiers.neural.common;

import java.io.Serializable;

import weka.core.Instance;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 * @author Jason Brownlee
 * @version 1.0
 */

public class SimpleNeuron implements Serializable
{
    public final static double DEFAULT_BIAS_VALUE = 1.0;
    
	protected final double biasInputValue;

    // weights to apply to inputs (num inputs + 1 for the bias)
    protected final double [] inputWeights;

    // derivatives of error in regard to weights
    protected final double [] dEwE;

    // last change in each weight
    protected final double [] lastWeightDeltas;

    // index in the weight vector of the bias weight (always at the end of the array)
    protected final int biasIndex;

	

    public SimpleNeuron(int numInputs, double aBiasInput)
    {
		biasInputValue = aBiasInput;
    	
        // +1 for the bias value (end)
        inputWeights     = new double[numInputs + 1];
        dEwE             = new double[numInputs + 1];
        lastWeightDeltas = new double[numInputs + 1];

        biasIndex = numInputs; // the end of the array
    }



    public double activate(Instance instance)
    {
        // calculates the activation given an instance

        double result = 0.0;
        double [] input = instance.toDoubleArray();
        int offset = 0;

        for(int i=0; i<input.length; i++)
        {
            // class values are not included
            if(i != instance.classIndex())
            {
                // never add missing values into the activation
                if(instance.isMissing(i))
                {
                    offset++;
                }
                else
                {
                    result += (input[i] * inputWeights[offset++]);
                }
            }
        }

        // add the bias output
        result += (biasInputValue * inputWeights[biasIndex]);

        return result;
    }

    public double activate(double [] inputs)
    {
        // calculate the activation given an input vector

        double result = 0.0;

        for(int i=0; i<inputs.length; i++)
        {
            result += (inputs[i] * inputWeights[i]);
        }

        // add the bias output
        result += (biasInputValue * inputWeights[biasIndex]);

        return result;
    }

    public double [] getdEwE()
    {
        return dEwE;
    }

    public double [] getLastWeightDeltas()
    {
        return lastWeightDeltas;
    }

    public double [] getWeights()
    {
        return inputWeights;
    }

    public int getBiasIndex()
    {
        return biasIndex;
    }
	/**
	 * @return
	 */
	public double getBiasInputValue()
	{
		return biasInputValue;
	}

}