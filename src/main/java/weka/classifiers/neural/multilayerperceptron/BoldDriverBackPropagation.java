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
import weka.classifiers.neural.multilayerperceptron.algorithm.BoldDriverBackPropagationAlgorithm;
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

public class BoldDriverBackPropagation extends WekaAlgorithmAncestor
{
    public final static int PARAM_TRANSFER_FUNCTION       = 0;
    public final static int PARAM_MOMENTUM                = 1;
    public final static int PARAM_WEIGHT_DECAY            = 2;
    public final static int PARAM_HIDDEN_1                = 3;
    public final static int PARAM_HIDDEN_2                = 4;
    public final static int PARAM_HIDDEN_3                = 5;

    public final static int PARAM_ERROR_INCREASE          = 6;
    public final static int PARAM_ERROR_DECREASE          = 7;
    
	public final static int PARAM_TRAINING_MODE           = 8;


    // param flags
    public final static String [] EXTRA_PARAMETERS =
    {
        "F", // transfer function
        "A", // momentum
        "D",  // weight decay
        "X",  // hidden layer 1 num nodes
        "Y",  // hidden layer 2 num nodes
        "Z",  // hidden layer 3 num nodes
        "K", // error increase
        "G", // error decrease
		"N" // training mode
    };
    
	public final static String [] EXTRA_PARAMETER_NOTES =
	{
		"<transfer function>", // transfer function
		"<momentum value>", // momentum		
		"<weight decay value>",  // weight decay
		"<total first layer nodes>",  // hidden layer 1 num nodes
		"<total second layer nodes>",  // hidden layer 2 num nodes
		"<total third layer nodes>",  // hidden layer 3 num nodes
		"<error learning rate increase>", // error increase
		"<error learning rate decrease>", // error decrease
		"<training mode>" // training mode
	};     

    // descriptions for all parameters
    public final static String [] EXTRA_PARAM_DESCRIPTIONS =
    {
        "Neuron transfer function "+TransferFunctionFactory.DESCRIPTION,
        "Momentum Factor (recommend between 0.0 and 0.9, 0.0==not used)",
        "Weight Decay Factor (recommend between 0.0 and 1.0, 0.0==not used)",
        "The number of nodes in the first hidden layer (0 for none)",
        "The number of nodes in the second hidden layer (0 for none)",
        "The number of nodes in the third hidden layer (0 for none)",
        "Scale factor to decrease the learning rate when global error increases (recommend 0.5)",
        "Scale factor to increase the learning rate when global error decreases (recommend 1.05)",
		"Model training algorithm "+TrainerFactory.DESCRIPTION,
    };


    // momentum
    protected double momentum = 0.0;

    // weight decay
    protected double weightDecay = 0.0;

    // topology
    protected int hiddenLayer1 = 0;
    protected int hiddenLayer2 = 0;
    protected int hiddenLayer3 = 0;


    protected double errorIncreaseAdjustment = 0.0;
    protected double errorDecreaseAdjustment = 0.0;




    public BoldDriverBackPropagation()
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

        errorIncreaseAdjustment = 0.5;
        errorDecreaseAdjustment = 1.05;
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
		BoldDriverBackPropagationAlgorithm algorithm = new BoldDriverBackPropagationAlgorithm(function, rand, lrateFunction, momentum, weightDecay, biasInput, hiddenLayersTopology, instances, errorIncreaseAdjustment, errorDecreaseAdjustment);

        return algorithm;

    }

    protected Collection getAlgorithmOptions()
    {
        ArrayList list = new ArrayList(14);

        list.add("-"+EXTRA_PARAMETERS[PARAM_TRANSFER_FUNCTION]);
        list.add(Integer.toString(transferFunction));

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
        
		list.add("-"+EXTRA_PARAMETERS[PARAM_TRAINING_MODE]);
		list.add(Integer.toString(trainingMode));


        list.add("-"+EXTRA_PARAMETERS[PARAM_ERROR_INCREASE]);
        list.add(Double.toString(errorIncreaseAdjustment));

        list.add("-"+EXTRA_PARAMETERS[PARAM_ERROR_DECREASE]);
        list.add(Double.toString(errorDecreaseAdjustment));


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

        buffer.append("Back Propagation Learning Rule with Bold Driver (Vogl's Method) adaptive learning rate algorithm");

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
				case PARAM_ERROR_INCREASE:
				{
					errorIncreaseAdjustment = Double.parseDouble(data);   
					break;
				}
				case PARAM_ERROR_DECREASE:
				{
					errorDecreaseAdjustment = Double.parseDouble(data);
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

    public String errorIncreaseAdjustmentTipText()
    {
        return EXTRA_PARAM_DESCRIPTIONS[PARAM_ERROR_INCREASE];
    }
    public String errorDecreaseAdjustmentTipText()
    {
        return EXTRA_PARAM_DESCRIPTIONS[PARAM_ERROR_DECREASE];
    }
	public String trainingModeTipText()
	{
		return EXTRA_PARAM_DESCRIPTIONS[PARAM_TRAINING_MODE];
	}


    public double getErrorIncreaseAdjustment()
    {
        return errorIncreaseAdjustment;
    }
    public void setErrorIncreaseAdjustment(double m)
    {
        errorIncreaseAdjustment = m;
    }


    public double getErrorDecreaseAdjustment()
    {
        return errorDecreaseAdjustment;
    }
    public void setErrorDecreaseAdjustment(double m)
    {
        errorDecreaseAdjustment = m;
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
			System.out.println(Evaluation.evaluateModel(new BoldDriverBackPropagation(), args));
		}
		catch (Exception e)
		{
			System.out.println(e.getMessage());
		}
	}    
}