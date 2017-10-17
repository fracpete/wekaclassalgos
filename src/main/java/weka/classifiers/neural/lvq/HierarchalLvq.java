package weka.classifiers.neural.lvq;

import java.text.NumberFormat;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.neural.lvq.initialise.InitialisationFactory;
import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.UnsupportedClassTypeException;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 * Date: 22/05/2004
 * File: HierarchalLVQ.java
 * 
 * @author Jason Brownlee
 *
 */
public class HierarchalLvq extends Classifier
	implements OptionHandler, WeightedInstancesHandler
{
	/**
	 * Index of the initialsiation parameter
	 */
	private final static int PARAM_BASE_ALGORITHM       = 0;
	/**
	 * Index of the code book parameter
	 */
	private final static int PARAM_SUB_CLASSIFIER      = 1;
	/**
	 * Index of the training iterations parameter
	 */
	private final static int PARAM_ERROR_PERCENTAGE    = 2;
	/**
	 * Index of the learning function parameter
	 */
	private final static int PARAM_HIT_PERCENTAGE      = 3;	
	/**
	 * Common LVQ algorithm parameters
	 */
	private final static String [] PARAMETERS =
	{
		"B", // base lvq algorithm
		"S", // sub algorithm
		"E", // error percentage
		"H" // hit percentage
	};
	private final static String [] PARAMETER_NOTES =
	{
		"<base LVQ algorithm>", // base lvq algorithm
		"<sub model algorithm>", // sub algorithm
		"<bmu error percentage>", // error percentage
		"<bmu hit percentage>" // hit percentage
	};	
	/**
	 * Descriptions of common LVQ algorithm parameters
	 */
	private final static String [] PARAM_DESCRIPTIONS =
	{
		"LVQ algorthm to construt the base LVQ model.",
		"Algorithm to use to construct the bmu sub models.",
		"Percentage of training error a bmu must achieve to be considered a candidate for a sub model.",
		"Percentage of total training hits a bmu must achieve to be considered a candidate for a sub model."
		
	};	
	/**
	 * Total number of classes in dataset
	 */
	protected int numClasses;
	/**
	 * Total number of attributes in dataset
	 */
	protected int numAttributes;
	/**
	 * Base LVQ model used for clustering
	 */
	private AlgorithmAncestor baseAlgorithm;
	/**
	 * Type and configuration of classifier to use for sub models
	 */
	private Classifier subModelType;
	/**
	 * Percentage of data running through a bmu for it to be considered a cluster
	 */
	private double hitPercentage;
	/**
	 * Percentage of error a bmu must exibit to be considered a cluster
	 */
	private double errorPercentage;
	/**
	 * Collection of sub models used instead of BMU's, indexed on bmu id
	 */
	private Classifier [] subModels;
	/**
	 * Training data used for training sub models, indexed on bmu id
	 */
	private Instances [] subModelTrainingData;
	/**
	 * Whether or not a sub model is used for each bmu id
	 */
	private boolean [] subModelUsed;	
	/**
	 * Matrix of bmu usage calculated after base model construction, using training data
	 */
	private int [][] trainingBmuUsage;
	/**
	 * Total number of bmu hits (sum of training bmu matrix)
	 */
	private long totalTrainingBmuHits;
	/**
	 * Accuracy of sub models on training data
	 */
	private double [] subModelAccuracy;
	/**
	 * Formatter used for producing useful information to the user
	 */
	private final static NumberFormat format = NumberFormat.getPercentInstance();
	
	
	public HierarchalLvq()
	{
		// prepare defaults
		baseAlgorithm = new Olvq1();
		baseAlgorithm.setInitialisationMode(new SelectedTag(InitialisationFactory.INITALISE_TRAINING_EVEN, InitialisationFactory.TAGS_MODEL_INITALISATION));
		subModelType = new MultipassLvq();
		hitPercentage = 1.0;
		errorPercentage = 10.0;
	}

	private void evaluateAndPruneClassifiers()
		throws Exception
	{
		for (int i = 0; i < subModelUsed.length; i++)
		{
			if(subModelUsed[i])			
			{
				// determine bmu's quality
				double bmuQuality = calculateBmuQuality(i);
				// determine the sub-model's quality				
				subModelAccuracy[i] = calculateSubModelQuality(i);
				// check if the quality of the sub model is worse than the bmu
				if(bmuQuality >= subModelAccuracy[i])
				{
					// prune the sub-model
					subModelUsed[i] = false;
				}
				// else keep the sub-model				
			}
		}
	}
	private String prepareSubModelAccuracyReport()
	{
		StringBuffer buffer = new StringBuffer(1024);
		
		buffer.append("-- BMU Sub-Model Accuracy --\n");
		buffer.append("bmu,\t%bmu,\t%model,\t%better,\tpruned\n");
		
		for (int i = 0; i < subModelUsed.length; i++)
		{			
			if(isBmuIdCandidate(i))		
			{
				// determine bmu's quality
				double bmuQuality = calculateBmuQuality(i);
				double improvement = subModelAccuracy[i] - bmuQuality;
				
				buffer.append(i);
				buffer.append(",\t");
				buffer.append(format.format(bmuQuality/100.0));
				buffer.append(",\t");
				buffer.append(format.format(subModelAccuracy[i]/100.0));
				buffer.append(",\t");
				buffer.append(format.format(improvement/100.0));
				buffer.append(",\t\t");
				if(!subModelUsed[i])
				{
					buffer.append("true");	
				}
				buffer.append("\n");
			}
		}		
		
		return buffer.toString();
	}
	private double calculateSubModelQuality(int aBmuId)
		throws Exception
	{
		Evaluation eval = new Evaluation(subModelTrainingData[aBmuId]);
		eval.evaluateModel(subModels[aBmuId], subModelTrainingData[aBmuId]);
		return eval.pctCorrect();
	}
	private double calculateBmuQuality(int aBmuId)
	{
		int totalHits = (trainingBmuUsage[aBmuId][0] + trainingBmuUsage[aBmuId][1]);
		
		// check for a sum hits of zero
		if(totalHits == 0)
		{
			return 0.0;
		}
		
		// total / total possible
		double percentCorrect = ((double)trainingBmuUsage[aBmuId][0] / (double)totalHits);
		// make useable
		percentCorrect *= 100.0;
		return percentCorrect;		
	}
	private void prepareClassifiersForCandidateClusters(Instances trainingDataset)
		throws Exception
	{
		for (int i = 0; i < subModelUsed.length; i++)
		{
			if(subModelUsed[i])
			{
				// initialise and train the classifier
				subModels[i] = prepareClusterClassifier(subModelTrainingData[i], trainingDataset, i);
			}			
		}
	}	
	private Classifier prepareClusterClassifier(Instances aClusterTrainingSet, Instances aTrainingSet, int aClusterNumber)
			throws Exception
	{
		// clone the selected model type		
		Classifier clusterClassifier = Classifier.makeCopies(subModelType, 1)[0];		
		// train the model
		clusterClassifier.buildClassifier(aClusterTrainingSet);
		return clusterClassifier;
	}	
	private void prepareDataForCandidateClusters(Instances trainingDataset)
	{
		// sort all training data by bmu
		LinkedList [] tmpList = new LinkedList[subModelTrainingData.length];
		for (int i = 0; i < trainingDataset.numInstances(); i++)
		{
			Instance instance = trainingDataset.instance(i);
			CodebookVector codebook = baseAlgorithm.getModel().getBmu(instance);
			int id = codebook.getId();
			if(tmpList[id] == null)
			{
				tmpList[id] = new LinkedList();
			}
			tmpList[id].add(instance);
		}	
		// convert datasets for known clusters into usable training data
		for (int i = 0; i < subModelUsed.length; i++)
		{
			if(subModelUsed[i])	
			{
				// check for no data in cluster
				if(tmpList[i] == null || tmpList[i].isEmpty())
				{
					subModelUsed[i] = false;
				}
				else
				{
					subModelTrainingData[i] = linkedListToInstances(tmpList[i], trainingDataset);
				}				
			}
			tmpList[i] = null; // reduce memory on the fly
		}
	}	
	private Instances linkedListToInstances(LinkedList aListOfInstances, Instances aInstances)
	{
		Instances instances = new Instances(aInstances, aListOfInstances.size());
		for (Iterator iter = aListOfInstances.iterator(); iter.hasNext();)
		{
			Instance element = (Instance) iter.next();
			instances.add(element);
		}
		return instances;		
	}	
	private void selectBMUCandidateClusters()
	{		
		for (int i = 0; i < trainingBmuUsage.length; i++)
		{		
			if(isBmuIdCandidate(i))
			{
				subModelUsed[i] = true;	
			}			
		}
	}
	private boolean isBmuIdCandidate(int aBmuId)
	{
		double error = getBmusPercentageError(aBmuId);
		double hits = getBmusHitPercentage(aBmuId);
		return isCandidate(error, hits);
	}
	private boolean isCandidate(double error, double hits)
	{
		// must have > n% of total hits
		if(hits >= hitPercentage)
		{
			// must have >= n% of hits are error
			if(error >= errorPercentage)
			{
				return true;
			}
		}
		
		return false;
	}
	private String prepareSubModelSelectionReport()
	{
		StringBuffer buffer = new StringBuffer(1024);
		
		buffer.append("-- BMU Sub-Model Selection Report --\n");
		buffer.append("bmu,\t%error,\t%hits,\tpruned\n");
		
		for (int i = 0; i < trainingBmuUsage.length; i++)
		{
			double error = getBmusPercentageError(i);
			double hits = getBmusHitPercentage(i);
			
			if(isCandidate(error, hits))
			{
				buffer.append(i);
				buffer.append(",\t");
				buffer.append(format.format(error/100.0));
				buffer.append(",\t");
				buffer.append(format.format(hits/100.0));
				buffer.append(",\t");					
				// model no longer exists
				if(!subModelUsed[i])
				{
					buffer.append("true");
				}				
				buffer.append("\n");
			}			
		}
		
		return buffer.toString();
	}
	private double getBmusHitPercentage(int aBmuId)
	{
		int totalHits = (trainingBmuUsage[aBmuId][0] + trainingBmuUsage[aBmuId][1]);
		
		// check for a sum hits of zero
		if(totalHits == 0)
		{
			return 0.0;
		}
		
		// total / total possible
		double percentHits = ((double)totalHits / (double)totalTrainingBmuHits);
		// make useable
		percentHits *= 100.0;
		return percentHits;
	}	
	private double getBmusPercentageError(int aBmuId)
	{
		int totalHits = (trainingBmuUsage[aBmuId][0] + trainingBmuUsage[aBmuId][1]);
		
		// check for a sum hits of zero
		if(totalHits == 0)
		{
			return 0.0;
		}
		
		// error / bmu hits
		double error = ((double)trainingBmuUsage[aBmuId][1] / (double)totalHits);
		// make useable
		error *= 100.0;
		return error;
	}
	private void getBmuHits(Instances trainingDataset)
		throws Exception
	{
		trainingBmuUsage = baseAlgorithm.getTrainingBmuUsage();
		totalTrainingBmuHits = baseAlgorithm.getTotalTrainingBmuHits();		
	}
	private void cleanup()
	{
		// training data is no longer needed
		subModelTrainingData = null;
	}
	private void initialiseBaseModel()
	{
		int totalCodebookVectors = baseAlgorithm.getTotalCodebookVectors();	
		baseAlgorithm.setDebug(m_Debug);
		// prepare other bits 
		subModelTrainingData = new Instances[totalCodebookVectors];
		subModels = new Classifier[totalCodebookVectors];
		subModelUsed = new boolean[totalCodebookVectors];	
		subModelAccuracy = new double[totalCodebookVectors];
	}
	
	
	/**
	 * Calcualte the class distribution for the provided instance
	 * @param instance - an instance to calculate the class distribution for
	 * @return double [] - class distribution for instance - all values are 0, exception for the 
	 *  index of the predicted class, which has the value of 1
	 * @throws Exception
	 */
	public double [] distributionForInstance(Instance instance)
		throws Exception
	{
		if(baseAlgorithm == null)
		{
			throw new Exception("Model has not been prepared");
		}
		// verify number of classes
		else if(instance.numClasses() != numClasses)
		{
			throw new Exception("Number of classes in instance ("+instance.numClasses()+") does not match expected ("+numClasses+").");
		}
		// verify the number of attributes
		else if(instance.numAttributes() != numAttributes)
		{
			throw new Exception("Number of attributes in instance (" + instance.numAttributes() + ") does not match expected (" + numAttributes + ").");
		}
		
		// get the bmu
		double [] classDistribution = null;
		CodebookVector bmu = baseAlgorithm.getModel().getBmu(instance);
		// check if the bmu is used for classification
		if(!subModelUsed[bmu.getId()])
		{
			classDistribution = new double[numClasses];

			// there is no class distribution, only the predicted class
			if(baseAlgorithm.getUseVoting())
			{
				// return the class distribution
				int [] distribution = bmu.getClassHitDistribution();
				int total = 0;
				// calculate the total hits
				for (int i = 0; i < distribution.length; i++)
				{
					total += distribution[i];
				}
				// calculate percentages for each class
				for (int i = 0; i < classDistribution.length; i++)
				{
					classDistribution[i] = ((double)distribution[i] / (double)total);
				}
			}
			else
			{
				int index = (int) bmu.getClassification();
				classDistribution[index] = 1.0;
			}			
		}
		// use the sub model
		else
		{			
			classDistribution = subModels[bmu.getId()].distributionForInstance(instance);
		}

		return classDistribution;
	}
	/**
	 * Build a model of the provided training dataset using the specific LVQ
	 * algorithm implementation. The model is constructed (if not already provided),
	 * it is initialised, then the model is trained (constructed) using
	 * the specific implementation of the LVQ algorithm by calling
	 * prepareLVQClassifier() 
	 * @param instances - training dataset.
	 * @throws Exception
	 */
	public void buildClassifier(Instances instances)
		throws Exception
	{
		// validate user provided arguments
		validateAlgorithmArguments();
		// prepare the dataset
		Instances trainingDataset = prepareDataset(instances);
		// prepare elements based on base model
		initialiseBaseModel();		
		// build the base model
		baseAlgorithm.buildClassifier(trainingDataset);		
		// extract bmu usage
		getBmuHits(trainingDataset);
		// select bmu candidate clusters
		selectBMUCandidateClusters();
		// prepare data for candidate clusters
		prepareDataForCandidateClusters(trainingDataset);
		// prepre classifiers for candidate clusters
		prepareClassifiersForCandidateClusters(trainingDataset);
		// evaluate candidate clusters, prune models that perform worse than BMU		
		evaluateAndPruneClassifiers();
		// clean up data no longer needed
		cleanup();
	}    
	/**
	 * Verify the dataset can be used with the LVQ algorithm and store details about
	 * the nature of the data.<br> 
	 * Rules:
	 * <ul>
	 * <li>Class must be assigned</li>
	 * <li>Class must be nominal</li>
	 * <li>Must be atleast 1 training instance</li>
	 * <li>Must have attributes besides the class attribute</li> 
	 * </ul>
	 * 
	 * @param instances - training dataset
	 * @return - all instances that can be used for training
	 * @throws Exception
	 */
	protected Instances prepareDataset(final Instances instances)
		throws Exception
	{       
		Instances trainingInstances = new Instances(instances);

		// must have a class assigned
		if (trainingInstances.classIndex() < 0)
		{
			throw new Exception("No class attribute assigned to instances");
		}
		// class must be nominal
		else if(!trainingInstances.classAttribute().isNominal())
		{
			throw new UnsupportedClassTypeException("Class attribute must be nominal");
		}
		// must have some training instances
		else if (trainingInstances.numInstances() == 0)
		{
			throw new Exception("No usable training instances!");
		}
		// must have attributes besides the class attribute
		else if(trainingInstances.numAttributes() <= +1)
		{
			throw new Exception("Dataset contains no supported comparable attributes");
		}

		numClasses = trainingInstances.numClasses();
		numAttributes = trainingInstances.numAttributes();

		// return training instances
		return trainingInstances;
	}	
	protected void validateAlgorithmArguments() throws Exception
	{
		if(baseAlgorithm == null)
		{
			throw new Exception("An LVQ algorithm used to construct the base model must be specified.");
		}
		else if(subModelType == null)
		{
			throw new Exception("A algorithm used to construct sub models for bmu's must be specified.");
		}
		else if(errorPercentage < 0.0 || errorPercentage > 100.0)		
		{
			throw new Exception("Error percentage must be in the range of 0.0 to 100.0.");
		}
		else if(hitPercentage < 0.0 || hitPercentage > 100.0)		
		{
			throw new Exception("Hit percentage must be in the range of 0.0 to 100.0.");
		}
	}	
	public Enumeration listOptions()
	{
		Vector list = new Vector(PARAMETERS.length);
		for(int i=0; i<PARAMETERS.length; i++)
		{
			String param = "-"+PARAMETERS[i]+" "+PARAMETER_NOTES[i];
			list.add(new Option("\t"+PARAM_DESCRIPTIONS[i], PARAMETERS[i], 1, param));
		}
		return list.elements();
	}	
	public void setOptions(String [] options)
		throws Exception
	{
		for (int i = 0; i < PARAMETERS.length; i++)
		{
			String data = Utils.getOption(PARAMETERS[i].charAt(0), options);
			
			if(data == null || data.length()==0)
			{
				continue;
			}
			
			switch(i)
			{
				case PARAM_BASE_ALGORITHM:
				{
					setBaseLVQAlgorithm(prepareClassifierFromParameterString(data));
					break;
				}
				case PARAM_SUB_CLASSIFIER:
				{
					setSubModelAlgorithm(prepareClassifierFromParameterString(data));
					break;
				}		
				case PARAM_ERROR_PERCENTAGE:
				{
					errorPercentage = Double.parseDouble(data);
					break;
				}
				case PARAM_HIT_PERCENTAGE:
				{
					hitPercentage = Double.parseDouble(data);
					break;
				}		
				default:
				{
					throw new Exception("Invalid option offset: " + i);
				}
			}
		}		
	}	
	
	
	protected boolean hasValue(String aString)
	{
		return (aString!=null && aString.length()!=0);
	}
	
	
	
   public String [] getOptions()
   {
	   LinkedList list = new LinkedList();

	   list.add("-"+PARAMETERS[PARAM_BASE_ALGORITHM]);
	   list.add(getClassifierSpec(baseAlgorithm));

	   list.add("-"+PARAMETERS[PARAM_SUB_CLASSIFIER]);
	   list.add(getClassifierSpec(subModelType));

	   list.add("-"+PARAMETERS[PARAM_ERROR_PERCENTAGE]);
	   list.add(Double.toString(errorPercentage));

	   list.add("-"+PARAMETERS[PARAM_HIT_PERCENTAGE]);
	   list.add(Double.toString(hitPercentage));      

	   return (String [] ) list.toArray(new String[list.size()]);
   }	
   protected String getClassifierSpec(Classifier c)
   {
	   String name = c.getClass().getName();
	   String params = Utils.joinOptions(c.getOptions());
	   return  name + " " + params;
   }   
	/**
	 * Prepares the provided classifier from a string
	 * @param s
	 * @return - classifier instance
	 * @throws Exception
	 */
	private Classifier prepareClassifierFromParameterString(String s)
		throws Exception
	{
		String [] classifierSpec = null;
		String classifierName = null;

		// split the string into its componenets
		classifierSpec = Utils.splitOptions(s);

		// verify some componets were specified
		if (classifierSpec.length == 0)
		{
			throw new Exception("Invalid classifier specification string");
		}

		// copy the name, then clear it from the list (it will not be a valid param for itself)
		classifierName = classifierSpec[0];
		classifierSpec[0] = "";

		// consrtuct the classifier with its params
		return Classifier.forName(classifierName, classifierSpec);
	}	
	public String globalInfo()
	{
		StringBuffer buffer = new StringBuffer();
		
		buffer.append("Hierarchal version of the LVQ/SOM algorithm where per-bmu models are used in some cases. ");
		buffer.append("For those bmu's that perform poorly, a sub-model is created to handle all classification ");
		buffer.append("tasks for data instances that match onto that bmu. Firstly the base LVQ/SOM model is constructed ");
		buffer.append("then evaluated, bmu's that are candidates for sub models are identifed based on their hit percentages. ");
		buffer.append("Finally sub models for candidate bmu's are created and evaluated. Those sub-models that ");
		buffer.append("out-perform their parent bmu (on the training data) are kept. \nUnlimited nesting of LVQ models ");
		buffer.append("can be achieved by selecting the HierarchalLVQ algorthm as the sub model implementation.");
		
		return buffer.toString();
	}	
	public String toString()
	{
		StringBuffer buffer = new StringBuffer();		

		if(super.m_Debug)
		{
			// bmu hits report
			if(baseAlgorithm.prepareBmuStatistis)
			{
				buffer.append(baseAlgorithm.prepareTrainingBMUReport());
				buffer.append("\n");
			}
		
			// class distributions for each codebook vector
			buffer.append(baseAlgorithm.prepareIndividualClassDistributionReport());
			buffer.append("\n");				
			
			// quantisation error
			buffer.append(baseAlgorithm.quantisationErrorReport());
			buffer.append("\n");
			
			// codebook vectors
			buffer.append(baseAlgorithm.prepareCodebookVectorReport());
			buffer.append("\n");
		}		
		
		// sub model selections
		buffer.append(prepareSubModelSelectionReport());
		buffer.append("\n");
		
		// sub model accuracy
		buffer.append(prepareSubModelAccuracyReport());
		buffer.append("\n");
		
		// build times
		buffer.append(baseAlgorithm.prepareBuildTimeReport());
		buffer.append("\n");
		
		// distribution report
		buffer.append(baseAlgorithm.prepareClassDistributionReport("-- Cass Distribution --"));
		buffer.append("\n");			
		
			
		
		return buffer.toString();
	}

	public void setBaseLVQAlgorithm(Classifier aClassifier)
	{
		if(aClassifier instanceof AlgorithmAncestor)
		{
			baseAlgorithm = (AlgorithmAncestor) aClassifier;
		}
		else
		{
			throw new IllegalArgumentException("Base algorithm must be a single pass LVQ or single pass SOM algorithm.");
		}
	}	
	public Classifier getBaseLVQAlgorithm()
	{
		return baseAlgorithm;
	}	
	public void setSubModelAlgorithm(Classifier aClassifier)
	{
		subModelType = aClassifier;
	}
	public Classifier getSubModelAlgorithm()
	{
		return subModelType;
	}
	public void setErrorPercentage(double aPercentage)
	{
		errorPercentage = aPercentage;
	}
	public double getErrorPercentage()
	{
		return errorPercentage;
	}
	public void setHitPercentage(double aPercentage)
	{
		hitPercentage = aPercentage;
	}
	public double getHitPercentage()
	{
		return hitPercentage;
	}
	
	public String baseLVQAlgorithmTipText()
	{
		return PARAM_DESCRIPTIONS[PARAM_BASE_ALGORITHM];
	}
	public String subModelAlgorithmTipText()
	{
		return PARAM_DESCRIPTIONS[PARAM_SUB_CLASSIFIER];
	}
	public String errorPercentageTipText()
	{
		return PARAM_DESCRIPTIONS[PARAM_ERROR_PERCENTAGE];
	}
	public String hitPercentageTipText()
	{
		return PARAM_DESCRIPTIONS[PARAM_HIT_PERCENTAGE];
	}
	/**
	 * Entry point into the algorithm for direct usage
	 * @param args
	 */
	public static void main(String [] args)
	{
		try
		{
			System.out.println(Evaluation.evaluateModel(new HierarchalLvq(), args));
		}
		catch (Exception e)
		{
			System.out.println(e.getMessage());
		}
	}	
}
