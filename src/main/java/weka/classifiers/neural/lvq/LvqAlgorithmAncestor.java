package weka.classifiers.neural.lvq;

import weka.classifiers.neural.common.Constants;
import weka.classifiers.neural.common.learning.LearningKernelFactory;
import weka.classifiers.neural.lvq.initialise.InitialisationFactory;
import weka.classifiers.neural.lvq.initialise.ModelInitialiser;
import weka.classifiers.neural.lvq.model.LvqModel;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SelectedTag;
import weka.core.Utils;

import java.util.Collection;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Vector;


/**
 * Description: Represents a common ancestor for specific LVQ algorithm
 * implementations. Provides common functionality shared between all LVQ
 * implementations. Provides a framwork to be implemented by specific LVQ
 * implementations for consistent validation, model construction and
 * instance classification.
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public abstract class LvqAlgorithmAncestor extends AlgorithmAncestor {

  private final static int PARAM_INITIALISATION = 0;

  private final static int PARAM_CODEBOOK_VECTORS = 1;

  private final static int PARAM_TRAINING_ITERAITONS = 2;

  private final static int PARAM_LEARNING_FUNCTION = 3;

  private final static int PARAM_LEARNING_RATE = 4;

  private final static int PARAM_RANDOM_SEED = 5;

  private final static int PARAM_USE_VOTING = 6;


  private final static String[] PARAMETERS =
    {
      "M", // initialisation mode
      "C", // total codebook vectors
      "I", // training iterations
      "L", // learning function
      "R",  // learning rate
      "S", // random number seed
      "G" // use voting
    };

  private final static String[] PARAMETER_NOTES =
    {
      "<initialisation mode>", // initialisation mode
      "<total code book vectors>", // total codebook vectors
      "<total training iterations>", // training iterations
      "<learning function>", // learning function
      "<inital learning rate>",  // learning rate
      "<random number seed>", // random number seed
      "<use voting>" // use voting
    };

  /**
   * Descriptions of common LVQ algorithm parameters
   */
  private final static String[] PARAM_DESCRIPTIONS =
    {
      Constants.DESCRIPTION_INITIALISATION,
      Constants.DESCRIPTION_CODEBOOK_VECTORS,
      Constants.DESCRIPTION_TRAINING_ITERATIONS,
      Constants.DESCRIPTION_LEARNING_FUNCTION,
      Constants.DESCRIPTION_LEARNING_RATE,
      Constants.DESCRIPTION_RANDOM_SEED,
      Constants.DESCRIPTION_USE_VOTING
    };

  protected int totalCodebookVectors;

  protected int trainingIterations;

  protected int learningFunction;

  protected double learningRate;


  public LvqAlgorithmAncestor() {
    // set default values
    totalCodebookVectors = 20;
    initialisationMode = InitialisationFactory.INITALISE_TRAINING_PROPORTIONAL;
    useVoting = false;
    seed = 1;
    trainingIterations = (totalCodebookVectors * 50);
    learningFunction = LearningKernelFactory.LEARNING_FUNCTION_LINEAR;
    learningRate = 0.3;
  }


  protected abstract Collection getListOptions();

  protected abstract void setArguments(String[] options) throws Exception;

  protected abstract Collection getAlgorithmOptions();

  protected abstract void validateArguments() throws Exception;


  protected void initialiseModel(Instances instances) {
    // construct the model
    model = new LvqModel(totalCodebookVectors);
    // initalise the model
    ModelInitialiser modelInit = InitialisationFactory.factory(initialisationMode, random, instances, model.getTotalCodebookVectors());
    model.initialiseModel(modelInit);
  }


  /**
   * Validate common LVQ algorithm arguments, calls implementation specific validation
   */
  protected void validateAlgorithmArguments() throws Exception {
    if (totalCodebookVectors <= 1) {
      throw new Exception("Total codebook vectors must be > 1");
    }

    if (trainingIterations <= 0) {
      throw new Exception("Total training iterations must be > 0");
    }

    validateArguments();
  }

  /**
   * Provides a list of common algorithm options, as well as specific options
   *
   * @return Enumeration
   */
  public Enumeration listOptions() {
    Vector list = new Vector(PARAMETERS.length);
    for (int i = 0; i < PARAMETERS.length; i++) {
      String param = "-" + PARAMETERS[i] + " " + PARAMETER_NOTES[i];

      list.add(new Option("\t" + PARAM_DESCRIPTIONS[i], PARAMETERS[i], 1, param));
    }
    Collection c = getListOptions();
    if (c != null) {
      list.addAll(c);
    }
    return list.elements();
  }

  /**
   * Set algorithm options, common and specific
   *
   * @param options - list of options
   */
  public void setOptions(String[] options)
    throws Exception {
    for (int i = 0; i < PARAMETERS.length; i++) {
      String data = Utils.getOption(PARAMETERS[i].charAt(0), options);

      if (data == null || data.length() == 0) {
	continue;
      }

      switch (i) {
	case PARAM_INITIALISATION: {
	  initialisationMode = Integer.parseInt(data);
	  break;
	}
	case PARAM_CODEBOOK_VECTORS: {
	  totalCodebookVectors = Integer.parseInt(data);
	  break;
	}
	case PARAM_TRAINING_ITERAITONS: {
	  trainingIterations = Integer.parseInt(data);
	  break;
	}
	case PARAM_LEARNING_FUNCTION: {
	  learningFunction = Integer.parseInt(data);
	  break;
	}
	case PARAM_LEARNING_RATE: {
	  learningRate = Double.parseDouble(data);
	  break;
	}
	case PARAM_RANDOM_SEED: {
	  seed = Long.parseLong(data);
	  break;
	}
	case PARAM_USE_VOTING: {
	  useVoting = Boolean.valueOf(data).booleanValue();
	  break;
	}
	default: {
	  throw new Exception("Invalid option offset: " + i);
	}
      }
    }

    // see if the decendents can make use of these options
    setArguments(options);

  }

  protected boolean hasValue(String aString) {
    return (aString != null && aString.length() != 0);
  }


  /**
   * Returns a list of all common and specific algorithm options
   */
  public String[] getOptions() {
    LinkedList list = new LinkedList();

    list.add("-" + PARAMETERS[PARAM_INITIALISATION]);
    list.add(Integer.toString(initialisationMode));

    list.add("-" + PARAMETERS[PARAM_CODEBOOK_VECTORS]);
    list.add(Integer.toString(totalCodebookVectors));

    list.add("-" + PARAMETERS[PARAM_TRAINING_ITERAITONS]);
    list.add(Integer.toString(trainingIterations));

    list.add("-" + PARAMETERS[PARAM_LEARNING_FUNCTION]);
    list.add(Integer.toString(learningFunction));

    list.add("-" + PARAMETERS[PARAM_LEARNING_RATE]);
    list.add(Double.toString(learningRate));

    list.add("-" + PARAMETERS[PARAM_RANDOM_SEED]);
    list.add(Long.toString(seed));

    list.add("-" + PARAMETERS[PARAM_USE_VOTING]);
    list.add(Boolean.toString(useVoting));

    Collection c = getAlgorithmOptions();
    if (c != null) {
      list.addAll(c);
    }

    return (String[]) list.toArray(new String[list.size()]);
  }

  /**
   * Initialisation mode tip
   *
   * @return
   */
  public String initialisationModeTipText() {
    return PARAM_DESCRIPTIONS[PARAM_INITIALISATION];
  }

  /**
   * Codebook vectors tip
   *
   * @return
   */
  public String totalCodebookVectorsTipText() {
    return PARAM_DESCRIPTIONS[PARAM_CODEBOOK_VECTORS];
  }

  /**
   * Training iterations tip
   *
   * @return
   */
  public String totalTrainingIterationsTipText() {
    return PARAM_DESCRIPTIONS[PARAM_TRAINING_ITERAITONS];
  }

  /**
   * Learning function tip
   *
   * @return
   */
  public String learningFunctionTipText() {
    return PARAM_DESCRIPTIONS[PARAM_LEARNING_FUNCTION];
  }

  /**
   * Learning rate tip
   *
   * @return
   */
  public String learningRateTipText() {
    return PARAM_DESCRIPTIONS[PARAM_LEARNING_RATE];
  }

  /**
   * Random number seed
   *
   * @return
   */
  public String randomSeedTipText() {
    return PARAM_DESCRIPTIONS[PARAM_RANDOM_SEED];
  }

  public String useVotingTipText() {
    return PARAM_DESCRIPTIONS[PARAM_USE_VOTING];
  }

  /**
   * Set total training iterations
   *
   * @param t
   */
  public void setTotalTrainingIterations(int t) {
    trainingIterations = t;
  }

  /**
   * Return total training iterations
   *
   * @return
   */
  public int getTotalTrainingIterations() {
    return trainingIterations;
  }


  /**
   * Set learning functiom
   *
   * @param l
   */
  public void setLearningFunction(SelectedTag l) {
    if (l.getTags() == LearningKernelFactory.TAGS_LEARNING_FUNCTION) {
      learningFunction = l.getSelectedTag().getID();
    }
  }

  /**
   * Return the learning function
   *
   * @return
   */
  public SelectedTag getLearningFunction() {
    return new SelectedTag(learningFunction, LearningKernelFactory.TAGS_LEARNING_FUNCTION);
  }

  /**
   * Set the learning rate
   *
   * @param r
   */
  public void setLearningRate(double r) {
    learningRate = r;
  }

  /**
   * Return the learning rate
   *
   * @return
   */
  public double getLearningRate() {
    return learningRate;
  }

  /**
   * @return
   */
  public int getTotalCodebookVectors() {
    return totalCodebookVectors;
  }

  /**
   * @param i
   */
  public void setTotalCodebookVectors(int i) {
    totalCodebookVectors = i;
  }

}