package weka.classifiers.neural.singlelayerperceptron;

import java.util.ArrayList;
import java.util.Collection;

import weka.classifiers.Evaluation;
import weka.classifiers.neural.common.NeuralModel;
import weka.classifiers.neural.common.SimpleNeuron;
import weka.classifiers.neural.common.WekaAlgorithmAncestor;
import weka.classifiers.neural.common.learning.LearningKernelFactory;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.common.training.TrainerFactory;
import weka.classifiers.neural.common.transfer.TransferFunction;
import weka.classifiers.neural.common.transfer.TransferFunctionFactory;
import weka.classifiers.neural.singlelayerperceptron.algorithm.PerceptronAlgorithm;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SelectedTag;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 * @author Jason Brownlee
 * @version 1.0
 */

public class Perceptron extends WekaAlgorithmAncestor
{
    private final static int EXTRA_PARAM_LEARNING_RATE_FUNCTION = 0;

    private final static String [] EXTRA_PARAMETERS =
    {
        "M" // learning rate function
    };
    
	private final static String [] EXTRA_PARAMETER_NOTES =
	{
		"<learning rate function>" // learning rate function
	};

    // descriptions for all parameters
    private final static String [] EXTRA_PARAM_DESCRIPTIONS =
    {
		"Learning rate function to use while training, static is typically better "+LearningKernelFactory.DESCRIPTION
    };



    public Perceptron()
    {
        // set static values
        transferFunction = TransferFunctionFactory.TRANSFER_SIGN;
        trainingMode = TrainerFactory.TRAINER_ONLINE;

        // set good initial values
        trainingIterations = 500;
        biasInput = SimpleNeuron.DEFAULT_BIAS_VALUE;
        learningRate = 0.1;
        learningRateFunction = LearningKernelFactory.LEARNING_FUNCTION_STATIC;
        randomNumberSeed = 0;
    }


    protected Collection getAlgorithmOptions()
    {
        ArrayList list = new ArrayList(2);

        list.add("-"+EXTRA_PARAMETERS[EXTRA_PARAM_LEARNING_RATE_FUNCTION]);
        list.add(Integer.toString(learningRateFunction));

        return list;

    }
    protected Collection getListOptions()
    {
         ArrayList list = new ArrayList(1);

        for(int i=0; i<EXTRA_PARAMETERS.length; i++)
        {
        	String param = "-"+EXTRA_PARAMETERS[i]+" "+EXTRA_PARAMETER_NOTES[i];
            list.add(new Option("\t"+EXTRA_PARAM_DESCRIPTIONS[i], EXTRA_PARAMETERS[i], 1, param));
        }

        return list;
    }

    public String globalInfo()
    {
        StringBuffer buffer = new StringBuffer();

        buffer.append("Single Layer Perceptron : Perceptron Learning Rule, Binary inputs, Sign transfer function");

        return buffer.toString();
    }

    protected NeuralModel prepareAlgorithm(Instances instances) throws java.lang.Exception
    {
        // prepare the transfer function
        TransferFunction transferFunc = TransferFunctionFactory.factory(transferFunction);
        // prepare the learning rate function
        LearningRateKernel learningFunction = LearningKernelFactory.factory(learningRateFunction, learningRate, trainingIterations);

        // construct the algorithm
        PerceptronAlgorithm algorithm = new PerceptronAlgorithm(transferFunc, biasInput, rand, learningFunction, instances);

        return algorithm;
    }

    protected void validateArguments() throws java.lang.Exception
    {
        // do nothing
    }

    protected void setArguments(String [] options) 
    	throws Exception
    {		
		for (int i = 0; i < EXTRA_PARAMETERS.length; i++)
		{
			String data = weka.core.Utils.getOption(EXTRA_PARAMETERS[i].charAt(0), options);
			
			if(data == null || data.length()==0)
			{
				continue;
			}
			
			switch(i)
			{
				case EXTRA_PARAM_LEARNING_RATE_FUNCTION:
				{
					learningRateFunction = Integer.parseInt(data);
					break;
				}
				default:
				{
					throw new Exception("Invalid option offset: " + i);
				}
			}
		}
    }

    public String learningRateFunctionTipText()
    {
        return EXTRA_PARAM_DESCRIPTIONS[EXTRA_PARAM_LEARNING_RATE_FUNCTION];
    }

    public void setLearningRateFunction(SelectedTag l)
    {
        if (l.getTags() == LearningKernelFactory.TAGS_LEARNING_FUNCTION)
        {
            learningRateFunction = l.getSelectedTag().getID();
        }
    }

    public SelectedTag getLearningRateFunction()
    {
        return new SelectedTag(learningRateFunction, LearningKernelFactory.TAGS_LEARNING_FUNCTION);
    }

	/**
	 * Entry point into the algorithm for direct usage
	 * @param args
	 */
	public static void main(String [] args)
	{
		try
		{
			System.out.println(Evaluation.evaluateModel(new Perceptron(), args));
		}
		catch (Exception e)
		{
			System.out.println(e.getMessage());
		}
	}


}