package weka.classifiers.neural.multilayerperceptron;

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
import weka.classifiers.neural.multilayerperceptron.algorithm.BackPropagationAlgorithm;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SelectedTag;
import weka.core.Utils;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 * @author Jason Brownlee
 * @version 1.0
 */

public class BackPropagation extends WekaAlgorithmAncestor
{
    public final static int PARAM_TRANSFER_FUNCTION       = 0;
    public final static int PARAM_TRAINING_MODE           = 1;
    public final static int PARAM_MOMENTUM                = 2;
    public final static int PARAM_WEIGHT_DECAY            = 3;
    public final static int PARAM_HIDDEN_1                = 4;
    public final static int PARAM_HIDDEN_2                = 5;
    public final static int PARAM_HIDDEN_3                = 6;
    public final static int PARAM_LEARNING_RATE_FUNCTION  = 7;

    // param flags
    public final static String [] EXTRA_PARAMETERS =
    {
        "F", // transfer function
        "N", // training mode
        "A", // momentum
        "D",  // weight decay
        "X",  // hidden layer 1 num nodes
        "Y",  // hidden layer 2 num nodes
        "Z",  // hidden layer 3 num nodes
        "M" // learning rate function
    };
    
    
	public final static String [] EXTRA_PARAMETER_NOTES =
	{
		"<transfer function>", // transfer function
		"<training mode>", // training mode
		"<momentum value>", // momentum
		"<weight decay value>",  // weight decay
		"<total first layer nodes>",  // hidden layer 1 num nodes
		"<total second layer nodes>",  // hidden layer 2 num nodes
		"<total third layer nodes>",  // hidden layer 3 num nodes
		"<learning function>" // learning rate function
	};    

    // descriptions for all parameters
    public final static String [] EXTRA_PARAM_DESCRIPTIONS =
    {
        "Neuron transfer function "+TransferFunctionFactory.DESCRIPTION,
        "Model training algorithm "+TrainerFactory.DESCRIPTION,
        "Momentum Factor (recommend between 0.0 and 0.9, 0.0==not used)",
        "Weight Decay Factor (recommend between 0.0 and 1.0, 0.0==not used)",
        "The number of nodes in the first hidden layer (0 for none)",
        "The number of nodes in the second hidden layer (0 for none)",
        "The number of nodes in the third hidden layer (0 for none)",
        "Learning rate function to use while training, static is typically better "+LearningKernelFactory.DESCRIPTION
    };


    // momentum
    protected double momentum = 0.0;

    // weight decay
    protected double weightDecay = 0.0;

    // topology
    protected int hiddenLayer1 = 0;
    protected int hiddenLayer2 = 0;
    protected int hiddenLayer3 = 0;




    public BackPropagation()
    {
        // set good initial values
        transferFunction     = TransferFunctionFactory.TRANSFER_SIGMOID;
        trainingMode         = TrainerFactory.TRAINER_BATCH;
        trainingIterations   = 500;
        biasInput            = SimpleNeuron.DEFAULT_BIAS_VALUE;
        learningRate         = 0.1;
        learningRateFunction = LearningKernelFactory.LEARNING_FUNCTION_STATIC;
        randomNumberSeed     = 0;

        momentum     = 0.2;
        weightDecay  = 0.0;
        hiddenLayer1 = 0;
        hiddenLayer2 = 0;
        hiddenLayer3 = 0;
    }



    protected NeuralModel prepareAlgorithm(Instances instances) throws java.lang.Exception
    {
        int [] hiddenLayersTopology = null;

        // prepare the transfer function
        TransferFunction function = TransferFunctionFactory.factory(transferFunction);
        // prepare the learning rate function
        LearningRateKernel lrateFunction = LearningKernelFactory.factory(learningRateFunction, learningRate, trainingIterations);

        // prepare hidden layers topology
        if(hiddenLayer1 <= 0)
        {
            hiddenLayersTopology = null;
        }
        else if(hiddenLayer2 <= 0)
        {
            hiddenLayersTopology = new int[]{hiddenLayer1};
        }
        else if(hiddenLayer3 <= 0)
        {
            hiddenLayersTopology = new int[]{hiddenLayer1, hiddenLayer2};
        }
        else
        {
            // all three hidden layers were specified
            hiddenLayersTopology = new int[]{hiddenLayer1, hiddenLayer2, hiddenLayer3};
        }

        // construct the algorithm
        BackPropagationAlgorithm algorithm = new BackPropagationAlgorithm(function, rand, lrateFunction, momentum, weightDecay, biasInput, hiddenLayersTopology, instances);


        return algorithm;

    }

    protected Collection getAlgorithmOptions()
    {
        ArrayList list = new ArrayList(14);

        list.add("-"+EXTRA_PARAMETERS[PARAM_TRANSFER_FUNCTION]);
        list.add(Integer.toString(transferFunction));

        list.add("-"+EXTRA_PARAMETERS[PARAM_TRAINING_MODE]);
        list.add(Integer.toString(trainingMode));

        list.add("-"+EXTRA_PARAMETERS[PARAM_MOMENTUM]);
        list.add(Double.toString(momentum));

        list.add("-"+EXTRA_PARAMETERS[PARAM_WEIGHT_DECAY]);
        list.add(Double.toString(weightDecay));

        list.add("-"+EXTRA_PARAMETERS[PARAM_HIDDEN_1]);
        list.add(Integer.toString(hiddenLayer1));

        list.add("-"+EXTRA_PARAMETERS[PARAM_HIDDEN_2]);
        list.add(Integer.toString(hiddenLayer2));

        list.add("-"+EXTRA_PARAMETERS[PARAM_HIDDEN_3]);
        list.add(Integer.toString(hiddenLayer3));

        list.add("-"+EXTRA_PARAMETERS[PARAM_LEARNING_RATE_FUNCTION]);
        list.add(Integer.toString(learningRateFunction));

        return list;
    }

    protected Collection getListOptions()
    {
        ArrayList list = new ArrayList(7);

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

        buffer.append("Back Propagation Learning Rule, variable number of hidden layers (0-3)");

        return buffer.toString();
    }

    protected void validateArguments() throws java.lang.Exception
    {
        if(hiddenLayer1 < 0)
        {
            throw new Exception("There must be >=0 nodes in the first layer: " + hiddenLayer1);
        }
        else if (hiddenLayer2 < 0)
        {
            throw new Exception("There must be >=0 nodes in the second layer: " + hiddenLayer2);
        }
        else if (hiddenLayer3 < 0)
        {
            throw new Exception("There must be >=0 nodes in the third layer: " + hiddenLayer3);
        }
        // check for valid topology
        else if(hiddenLayer2>0 && hiddenLayer1==0)
        {
            throw new Exception("A first layer must be specified to allow a second layer");
        }
        else if(hiddenLayer3>0 && hiddenLayer2==0)
        {
            throw new Exception("A second layer must be specified to allow a third layer");
        }
    }

    protected void setArguments(String [] options) 
    	throws Exception
    {    	
		for (int i = 0; i < EXTRA_PARAMETERS.length; i++)
		{
			String data = Utils.getOption(EXTRA_PARAMETERS[i].charAt(0), options);
			
			if(data == null || data.length()==0)
			{
				continue;
			}
			
			switch(i)
			{
				case PARAM_TRANSFER_FUNCTION:
				{
					transferFunction = Integer.parseInt(data);
					break;
				}
				case PARAM_TRAINING_MODE:
				{
					trainingMode = Integer.parseInt(data);
					break;
				}
				case PARAM_MOMENTUM:
				{
					momentum = Double.parseDouble(data);
					break;
				}
				case PARAM_WEIGHT_DECAY:
				{
					weightDecay = Double.parseDouble(data);
					break;
				}
				case PARAM_HIDDEN_1:
				{
					hiddenLayer1 = Integer.parseInt(data);
					break;
				}
				case PARAM_HIDDEN_2:
				{
					hiddenLayer2 = Integer.parseInt(data);
					break;
				}
				case PARAM_HIDDEN_3:
				{
					hiddenLayer3 = Integer.parseInt(data);
					break;
				}
				case PARAM_LEARNING_RATE_FUNCTION:
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


    public String transferFunctionTipText()
    {
        return EXTRA_PARAM_DESCRIPTIONS[PARAM_TRANSFER_FUNCTION];
    }
    public String trainingModeTipText()
    {
        return EXTRA_PARAM_DESCRIPTIONS[PARAM_TRAINING_MODE];
    }
    public String momentumTipText()
    {
        return EXTRA_PARAM_DESCRIPTIONS[PARAM_MOMENTUM];
    }
    public String weightDecayTipText()
    {
        return EXTRA_PARAM_DESCRIPTIONS[PARAM_WEIGHT_DECAY];
    }
    public String hiddenLayer1TipText()
    {
        return EXTRA_PARAM_DESCRIPTIONS[PARAM_HIDDEN_1];
    }
    public String hiddenLayer2TipText()
    {
        return EXTRA_PARAM_DESCRIPTIONS[PARAM_HIDDEN_2];
    }
    public String hiddenLayer3TipText()
    {
        return EXTRA_PARAM_DESCRIPTIONS[PARAM_HIDDEN_3];
    }
    public String learningRateFunctionTipText()
    {
        return EXTRA_PARAM_DESCRIPTIONS[PARAM_LEARNING_RATE_FUNCTION];
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


    public void setTransferFunction(SelectedTag l)
    {
        if(l.getTags() == TransferFunctionFactory.TAGS_TRANSFER_FUNCTION)
        {
            transferFunction = l.getSelectedTag().getID();
        }
    }
    public SelectedTag getTransferFunction()
    {
        return new SelectedTag(transferFunction, TransferFunctionFactory.TAGS_TRANSFER_FUNCTION);
    }


    public void setTrainingMode(SelectedTag l)
    {
        if(l.getTags() == TrainerFactory.TAGS_TRAINING_MODE)
        {
            trainingMode = l.getSelectedTag().getID();
        }
    }
    public SelectedTag getTrainingMode()
    {
        return new SelectedTag(trainingMode, TrainerFactory.TAGS_TRAINING_MODE);
    }

    public double getMomentum()
    {
        return momentum;
    }
    public void setMomentum(double m)
    {
        momentum = m;
    }


    public double getWeightDecay()
    {
        return weightDecay;
    }
    public void setWeightDecay(double w)
    {
        weightDecay = w;
    }

    public int getHiddenLayer1()
    {
        return hiddenLayer1;
    }
    public void setHiddenLayer1(int h)
    {
        hiddenLayer1 = h;
    }

    public int getHiddenLayer2()
    {
        return hiddenLayer2;
    }
    public void setHiddenLayer2(int h)
    {
        hiddenLayer2 = h;
    }

    public int getHiddenLayer3()
    {
        return hiddenLayer3;
    }
    public void setHiddenLayer3(int h)
    {
        hiddenLayer3 = h;
    }

	/**
	 * Entry point into the algorithm for direct usage
	 * @param args
	 */
	public static void main(String [] args)
	{
		try
		{
			System.out.println(Evaluation.evaluateModel(new BackPropagation(), args));
		}
		catch (Exception e)
		{
			System.out.println(e.getMessage());
		}
	}
}