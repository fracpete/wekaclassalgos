package weka.classifiers.neural.common;

import java.util.Collection;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.neural.common.learning.LearningKernelFactory;
import weka.classifiers.neural.common.training.NeuralTrainer;
import weka.classifiers.neural.common.training.TrainerFactory;
import weka.classifiers.neural.common.transfer.TransferFunctionFactory;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.UnsupportedClassTypeException;
import weka.core.WeightedInstancesHandler;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 * @author Jason Brownlee
 * @version 1.0
 */

public abstract class WekaAlgorithmAncestor extends Classifier
    implements OptionHandler, WeightedInstancesHandler
{
    private final static int PARAM_TRAINING_ITERATIONS    = 0;
    private final static int PARAM_LEARNING_RATE          = 1;
    private final static int PARAM_BIAS_CONSTANT          = 2;
    private final static int PARAM_RANDOM_SEED            = 3;

    // param flags
    private final static String [] PARAMETERS =
    {
        "I", // iterations
        "L", // learning rate
        "B",  // bias constant
        "R"  // random seed
    };
    
	// param flags
	private final static String [] PARAMETER_NOTES =
	{
		"<total training iterations>", // iterations
		"<learning rate>", // learning rate
		"<bias constant value>",  // bias constant
		"<random number seed>"  // random seed
	};

    // descriptions for all parameters
    private final static String [] PARAM_DESCRIPTIONS =
    {
        "Number of training iterations (anywhere from a few hundred to a few thousand)",
        "Learning Rate - between 0.05 and 0.75 (recommend 0.1 for most cases)",
        "Bias constant input, (recommend 1.0, use 0.0 for no bias constant input)",
        Constants.DESCRIPTION_RANDOM_SEED
    };

    // the model
    protected NeuralModel model;
    
	protected RandomWrapper rand;

    // random number seed
    protected long randomNumberSeed = 0;

    // learning rate
    protected double learningRate = 0.0;

    // learning rate function
    protected int learningRateFunction = 0;

    // bias input constant
    protected double biasInput = 0.0;

    // transfer function
    protected int transferFunction = 0;

    // training mode
    protected int trainingMode = 0;

    // number of training iterations
    protected int trainingIterations = 0;

    // stats on the dataset used to build the model
    protected int numInstances = 0;
    protected int numClasses = 0;
    protected int numAttributes = 0;

    protected boolean classIsNominal = false;




    public    abstract String globalInfo();

    protected abstract void validateArguments() throws Exception;

    protected abstract NeuralModel prepareAlgorithm(Instances instances) throws Exception;

    protected abstract Collection getListOptions();

    protected abstract void setArguments(String [] options) throws Exception;

    protected abstract Collection getAlgorithmOptions();


	
	public double [] getAllWeights()
	{
		return model.getAllWeights();
	}


    public void buildClassifier(Instances instances)
        throws Exception
    {
        // prepare the random number seed
		rand = new RandomWrapper(randomNumberSeed);

        // prepare the dataset for use
        Instances trainingInstances = prepareTrainingDataset(instances);

        // whether or not the class is nominal
        if(trainingInstances.classAttribute().isNominal())
        {
            classIsNominal = true;
        }
        else
        {
            classIsNominal = false;
        }

        // validate user provided arguments
        validateAlgorithmArguments();

        // initialise the model
        model = prepareAlgorithm(trainingInstances);

        // build the model
        NeuralTrainer trainer = TrainerFactory.factory(trainingMode, rand);
        trainer.trainModel(model, trainingInstances, trainingIterations);
    }


    protected void validateAlgorithmArguments() throws Exception
    {
        // num training iterations
        if(trainingIterations <= 0)
        {
            throw new Exception("The number of training iterations must be > 0");
        }

        // validate child arguments
        validateArguments();
    }



    public double[] distributionForInstance(Instance instance)
        throws Exception
    {
        if(model == null)
        {
            throw new Exception("Model has not been constructed");
        }

        // verify number of classes
        if(instance.numClasses() != numClasses)
        {
            throw new Exception("Number of classes in instance ("+instance.numClasses()+") does not match expected ("+numClasses+").");
        }

        // verify the number of attributes
        if(instance.numAttributes() != numAttributes)
        {
            throw new Exception("Number of attributes in instance (" + instance.numAttributes() + ") does not match expected (" + numAttributes + ").");
        }

        // get the network output
        double [] output = model.getDistributionForInstance(instance);

        // return the class distribution
        return output;
    }



    protected Instances prepareTrainingDataset(Instances aInstances) throws Exception
    {
        Instances trainingInstances = new Instances(aInstances);

        // must have a class assigned
        if (trainingInstances.classIndex() < 0)
        {
            throw new Exception("No class attribute assigned to instances");
        }

        // must have attributes besides the class attribute
        if(trainingInstances.numAttributes() <= +1)
        {
            throw new Exception("Dataset contains no supported comparable attributes");
        }

        // class must be nominal or numeric
        if(!trainingInstances.classAttribute().isNominal() && !trainingInstances.classAttribute().isNumeric())
        {
            throw new UnsupportedClassTypeException("Class attribute must be nominal");
        }

        // check each attribute
        for(int i=0; i<trainingInstances.numAttributes(); i++)
        {
            // all non-class attributes must be numeric
            if(i != trainingInstances.classIndex())
            {
                if (!trainingInstances.attribute(i).isNumeric())
                {
                    throw new Exception("Only numeric attributes are supported as network inputs");
                }
            }
        }

        // remove instances with missing class values
        trainingInstances.deleteWithMissingClass();

        // must have some training instances
        if (trainingInstances.numInstances() == 0)
        {
            throw new Exception("No usable training instances!");
        }

        numInstances = trainingInstances.numInstances();
        numClasses = trainingInstances.numClasses();
        numAttributes = trainingInstances.numAttributes();

        return trainingInstances;
    }


    public String toString()
    {
        StringBuffer buffer = new StringBuffer(200);

        buffer.append("--------------------------------------------");

        buffer.append("\n");

        // algorithm name
        buffer.append(globalInfo() + "\n");

        // check if the model has been constructed
        if (model == null)
        {
            buffer.append("The model has not been constructed");
        }
        else
        {
            buffer.append("Random Number Seed:     " + randomNumberSeed + "\n");
            buffer.append("Learning Rate:          " + learningRate + "\n");
            buffer.append("Learning Rate Function: " + LearningKernelFactory.getDescription(learningRateFunction) + "\n");
            buffer.append("Constant Bias Input:    " + biasInput + "\n");
            buffer.append("Training Iterations:    " + trainingIterations + "\n");
            buffer.append("Training Mode:          " + TrainerFactory.getDescriptionForMode(trainingMode) + "\n");
            buffer.append("Transfer Function       " + TransferFunctionFactory.getDescriptionForFunction(transferFunction) + "\n");
            buffer.append("\n");
            buffer.append(model.getModelInformation());
        }

        buffer.append("--------------------------------------------");

        return buffer.toString();
    }


    public Enumeration listOptions()
    {
        Vector list = new Vector(PARAMETERS.length);

        for(int i=0; i<PARAMETERS.length; i++)
        {
        	String param = "-"+PARAMETERS[i]+" "+PARAMETER_NOTES[i];
            list.add(new Option("\t"+PARAM_DESCRIPTIONS[i], PARAMETERS[i], 1, param));
        }

        Collection c = getListOptions();
        if(c!=null)
        {
            list.addAll(c);
        }

        return list.elements();
    }

    public void setOptions(String [] options)
        throws Exception
    {
		String [] values = new String[PARAMETERS.length];		 
		for (int i = 0; i < values.length; i++)
		{
			values[i] = weka.core.Utils.getOption(PARAMETERS[i].charAt(0), options);
		}		
    	
		for (int i = 0; i < values.length; i++)
		{
			String data = values[i];
			
			if(data == null || data.length()==0)
			{
				continue;
			}
			
			switch(i)
			{
				case PARAM_TRAINING_ITERATIONS:
				{
					trainingIterations = Integer.parseInt(data);
					break;
				}
				case PARAM_LEARNING_RATE:
				{
					learningRate = Double.parseDouble(data);
					break;
				}		
				case PARAM_BIAS_CONSTANT:
				{
					biasInput = Double.parseDouble(data);
					break;
				}
				case PARAM_RANDOM_SEED:
				{
					randomNumberSeed = Long.parseLong(data);
					break;
				}		
				default:
				{
					throw new Exception("Invalid option offset: " + i);
				}
			}
		}
		
		// pass of options to decendents
		setArguments(options);
    }


	protected boolean hasValue(String aString)
	{
		return (aString!=null && aString.length()!=0);
	}


    public String [] getOptions()
    {
        LinkedList list = new LinkedList();

        list.add("-"+PARAMETERS[PARAM_TRAINING_ITERATIONS]);
        list.add(Integer.toString(trainingIterations));

        list.add("-"+PARAMETERS[PARAM_LEARNING_RATE]);
        list.add(Double.toString(learningRate));

        list.add("-"+PARAMETERS[PARAM_BIAS_CONSTANT]);
        list.add(Double.toString(biasInput));

        list.add("-"+PARAMETERS[PARAM_RANDOM_SEED]);
        list.add(Long.toString(randomNumberSeed));

        Collection c = getAlgorithmOptions();
        if(c!=null)
        {
            list.addAll(c);
        }

        return (String [] ) list.toArray(new String[list.size()]);
    }

    public String trainingIterationsTipText()
    {
        return PARAM_DESCRIPTIONS[PARAM_TRAINING_ITERATIONS];
    }

    public String learningRateTipText()
    {
        return PARAM_DESCRIPTIONS[PARAM_LEARNING_RATE];
    }

    public String biasInputTipText()
    {
        return PARAM_DESCRIPTIONS[PARAM_BIAS_CONSTANT];
    }

    public String randomNumberSeedTipText()
    {
        return PARAM_DESCRIPTIONS[PARAM_RANDOM_SEED];
    }





    // accessor and mutator for algorithm parameters
    public int getTrainingIterations()
    {
        return trainingIterations;
    }
    public void setTrainingIterations(int i)
    {
        trainingIterations = i;
    }

    public double getLearningRate()
    {
        return learningRate;
    }
    public void setLearningRate(double l)
    {
        learningRate = l;
    }

    public double getBiasInput()
    {
        return biasInput;
    }
    public void setBiasInput(double l)
    {
        biasInput = l;
    }

    public long getRandomNumberSeed()
    {
        return randomNumberSeed;
    }
    public void setRandomNumberSeed(long l)
    {
        randomNumberSeed = l;
    }
}