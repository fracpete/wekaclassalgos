/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package weka.classifiers.neural.lvq;

import weka.classifiers.Evaluation;
import weka.classifiers.neural.common.Constants;
import weka.classifiers.neural.common.learning.LearningKernelFactory;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.algorithm.SomAlgorithm;
import weka.classifiers.neural.lvq.initialise.InitialisationFactory;
import weka.classifiers.neural.lvq.initialise.ModelInitialiser;
import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.classifiers.neural.lvq.model.SomModel;
import weka.classifiers.neural.lvq.neighborhood.NeighbourhoodKernel;
import weka.classifiers.neural.lvq.neighborhood.NeighbourhoodKernelFactory;
import weka.classifiers.neural.lvq.topology.NeighbourhoodDistance;
import weka.classifiers.neural.lvq.topology.NeighbourhoodDistanceFactory;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SelectedTag;
import weka.core.Utils;

import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Vector;

/**
 * Date: 25/05/2004
 * File: SupervisedSom.java
 *
 * @author Jason Brownlee
 */
public class Som extends AlgorithmAncestor {

  private final static int PARAM_MAP_WIDTH = 0;

  private final static int PARAM_MAP_HEIGHT = 1;

  private final static int PARAM_MAP_TOPOLOGY = 2;

  private final static int PARAM_MAP_SEED = 3;

  private final static int PARAM_MAP_INIT_MODE = 4;

  private final static int PARAM_MAP_NEIGH_FUNC = 5;

  private final static int PARAM_MAP_LEARN_FUNC = 6;

  private final static int PARAM_MAP_NEIGH_SIZE = 7;

  private final static int PARAM_MAP_LEARN_RATE = 8;

  private final static int PARAM_MAP_TRAIN_ITER = 9;

  private final static int PARAM_MAP_SUPERVISED = 10;

  private final static int PARAM_MAP_VOTING = 11;

  private final static String[] PARAMETERS =
    {
      "W", // width
      "H", // height
      "M", // topology
      "R", // random seed
      "I", // init mode
      "N", // neigh func
      "L", // learning func
      "K", // neigh size
      "P", // learn rate
      "F",  // iterations
      "S", // supervised
      "V"  // voting
    };

  private final static String[] PARAMETER_NOTES =
    {
      "<map width>", // width
      "<map height>", // height
      "<map topology>", // topology
      "<random number seed>", // random seed
      "<initialisation mode>", // init mode
      "<neighbourhood function>", // neigh func
      "<learning function>", // learning func
      "<initial neighbourhood size>", // neigh size
      "<initial learning rate>", // learn rate
      "<total training iterations>",  // iterations
      "<use supervised>", // supervised
      "<use voting>"  // voting
    };

  private final static String[] PARAM_DESCRIPTIONS =
    {
      "Map width (should be larger than the height).",
      "Map height.",
      "Map toplogy, (hexagonal is typically better) " + NeighbourhoodDistanceFactory.DESCRIPTION,
      Constants.DESCRIPTION_RANDOM_SEED,
      Constants.DESCRIPTION_INITIALISATION,
      "Map neighbourhood function " + NeighbourhoodKernelFactory.DESCRIPTION,
      Constants.DESCRIPTION_LEARNING_FUNCTION,
      "Initial neighbourhood size, should be the maps largest dimension.",
      "Initial learning rate, typically smaller than values used in LVQ",
      Constants.DESCRIPTION_TRAINING_ITERATIONS,
      "Whether or not the map will be trained in a supervised (LVQ-SOM) or unsupervised (SOM) manner.",
      "Whether voting is used to dynamically determine codebook's desired class, otherwise map labelling is performed after training."

    };

  protected int mapWidth;

  protected int mapHeight;

  protected int mapTopology;

  protected int neighbourhoodFunction;

  protected int learningFunction;

  protected int neighbourhoodSize;

  protected double learningRate;

  protected int trainingIterations;

  protected boolean supervised;


  public Som() {
    // set defaults
    mapWidth = 8;
    mapHeight = 6;
    mapTopology = NeighbourhoodDistanceFactory.NEIGHBOURHOOD_DISTNACE_HEXAGONAL;
    initialisationMode = InitialisationFactory.INITALISE_TRAINING_PROPORTIONAL;
    neighbourhoodFunction = NeighbourhoodKernelFactory.NEIGHBOURHOOD_KERNEL_GAUSSIAN;
    learningFunction = LearningKernelFactory.LEARNING_FUNCTION_LINEAR;
    neighbourhoodSize = Math.max(mapWidth, mapHeight);
    learningRate = 0.3;
    trainingIterations = mapWidth * mapHeight * 30;
    supervised = false;
    useVoting = false;
    seed = 1;
  }


  protected void initialiseModel(Instances instances) {
    // construct the model
    NeighbourhoodDistance distance = NeighbourhoodDistanceFactory.factory(mapTopology);
    model = new SomModel(distance, mapWidth, mapHeight);
    // initalise the model
    ModelInitialiser modelInit = InitialisationFactory.factory(initialisationMode, random, instances, model.getTotalCodebookVectors());
    model.initialiseModel(modelInit);
  }

  protected void trainModel(Instances instances) {
    // train the model
    LearningRateKernel learningKernel = LearningKernelFactory.factory(learningFunction, learningRate, trainingIterations);
    NeighbourhoodKernel neighbourhoodKernel = NeighbourhoodKernelFactory.factory(neighbourhoodFunction, neighbourhoodSize, trainingIterations);
    SomAlgorithm algorithm = new SomAlgorithm(learningKernel, neighbourhoodKernel, (SomModel) model, random, supervised);
    // add event listeners
    addEventListenersToAlgorithm(algorithm);
    // train the model
    algorithm.trainModel(instances, trainingIterations);
    // check if the map needs labelling
    if (!useVoting) {
      // clear class distributions
      model.clearClassDistributions();
      // enable voting on the model
      model.setUseVoting(true);
      try {
	// label the map using voting
	algorithm.labelMap(instances);
      }
      catch (Exception e) {
	e.printStackTrace();
      }
    }
  }


  /**
   * Overriden
   */
  public double[] distributionForInstance(Instance instance)
    throws Exception {
    if (model == null) {
      throw new Exception("Model has not been prepared");
    }
    // verify number of classes
    else if (instance.numClasses() != numClasses) {
      throw new Exception("Number of classes in instance (" + instance.numClasses() + ") does not match expected (" + numClasses + ").");
    }
    // verify the number of attributes
    else if (instance.numAttributes() != numAttributes) {
      throw new Exception("Number of attributes in instance (" + instance.numAttributes() + ") does not match expected (" + numAttributes + ").");
    }

    // get the bmu for the instance
    CodebookVector bmu = model.getBmu(instance);
    double[] classDistribution = new double[numClasses];

    // return the class distribution
    int[] distribution = bmu.getClassHitDistribution();
    int total = 0;
    // calculate the total hits
    for (int i = 0; i < distribution.length; i++) {
      total += distribution[i];
    }
    // calculate percentages for each class
    for (int i = 0; i < classDistribution.length; i++) {
      classDistribution[i] = ((double) distribution[i] / (double) total);
    }

    return classDistribution;
  }


  public int getTotalCodebookVectors() {
    return (mapWidth * mapHeight);
  }

  protected void validateAlgorithmArguments() throws Exception {
    if (mapWidth <= 0) {
      throw new Exception("Map width must be > 0");
    }
    else if (mapHeight <= 0) {
      throw new Exception("Map height must be > 0");
    }
    else if (trainingIterations <= 0) {
      throw new Exception("The number of training iterations must be > 0");
    }
  }

  public Enumeration listOptions() {
    Vector list = new Vector(PARAMETERS.length);
    for (int i = 0; i < PARAMETERS.length; i++) {
      String param = "-" + PARAMETERS[i] + " " + PARAMETER_NOTES[i];
      list.add(new Option("\t" + PARAM_DESCRIPTIONS[i], PARAMETERS[i], 1, param));
    }
    return list.elements();
  }

  public void setOptions(String[] options)
    throws Exception {
    for (int i = 0; i < PARAMETERS.length; i++) {
      String data = Utils.getOption(PARAMETERS[i].charAt(0), options);

      if (data == null || data.length() == 0) {
	continue;
      }

      switch (i) {
	case PARAM_MAP_WIDTH: {
	  mapWidth = Integer.parseInt(data);
	  break;
	}
	case PARAM_MAP_HEIGHT: {
	  mapHeight = Integer.parseInt(data);
	  break;
	}
	case PARAM_MAP_TOPOLOGY: {
	  mapTopology = Integer.parseInt(data);
	  break;
	}
	case PARAM_MAP_SEED: {
	  seed = Long.parseLong(data);
	  break;
	}
	case PARAM_MAP_INIT_MODE: {
	  initialisationMode = Integer.parseInt(data);
	  break;
	}
	case PARAM_MAP_NEIGH_FUNC: {
	  neighbourhoodFunction = Integer.parseInt(data);
	  break;
	}
	case PARAM_MAP_LEARN_FUNC: {
	  learningFunction = Integer.parseInt(data);
	  break;
	}
	case PARAM_MAP_NEIGH_SIZE: {
	  neighbourhoodSize = Integer.parseInt(data);
	  break;
	}
	case PARAM_MAP_LEARN_RATE: {
	  learningRate = Double.parseDouble(data);
	  break;
	}
	case PARAM_MAP_TRAIN_ITER: {
	  trainingIterations = Integer.parseInt(data);
	  break;
	}
	case PARAM_MAP_SUPERVISED: {
	  supervised = Boolean.valueOf(data).booleanValue();
	  break;
	}
	case PARAM_MAP_VOTING: {
	  useVoting = Boolean.valueOf(data).booleanValue();
	  break;
	}
	default: {
	  throw new Exception("Invalid option offset: " + i);
	}
      }
    }
  }

  public String[] getOptions() {
    LinkedList list = new LinkedList();

    list.add("-" + PARAMETERS[PARAM_MAP_WIDTH]);
    list.add("" + mapWidth);
    list.add("-" + PARAMETERS[PARAM_MAP_HEIGHT]);
    list.add("" + mapHeight);
    list.add("-" + PARAMETERS[PARAM_MAP_TOPOLOGY]);
    list.add("" + mapTopology);
    list.add("-" + PARAMETERS[PARAM_MAP_SEED]);
    list.add("" + seed);
    list.add("-" + PARAMETERS[PARAM_MAP_INIT_MODE]);
    list.add("" + initialisationMode);
    list.add("-" + PARAMETERS[PARAM_MAP_NEIGH_FUNC]);
    list.add("" + neighbourhoodFunction);
    list.add("-" + PARAMETERS[PARAM_MAP_LEARN_FUNC]);
    list.add("" + learningFunction);
    list.add("-" + PARAMETERS[PARAM_MAP_NEIGH_SIZE]);
    list.add("" + neighbourhoodSize);
    list.add("-" + PARAMETERS[PARAM_MAP_LEARN_RATE]);
    list.add("" + learningRate);
    list.add("-" + PARAMETERS[PARAM_MAP_TRAIN_ITER]);
    list.add("" + trainingIterations);
    list.add("-" + PARAMETERS[PARAM_MAP_SUPERVISED]);
    list.add("" + supervised);
    list.add("-" + PARAMETERS[PARAM_MAP_VOTING]);
    list.add("" + useVoting);

    return (String[]) list.toArray(new String[list.size()]);
  }


  public String globalInfo() {
    StringBuffer buffer = new StringBuffer();

    buffer.append("Self Organising Map (SOM), aka Kohonen Feature Map. ");
    buffer.append("The SOM algorithm is not intended to be used for classification, ");
    buffer.append("this is a version of the SOM that supports supervised learning, as well ");
    buffer.append("as unsupervised learning. Class labels can be assigned either dynamically via voting during ");
    buffer.append("training, or codebook vector labeling using after training.");

    return buffer.toString();
  }

  /**
   * Entry point into the algorithm for direct usage
   *
   * @param args
   */
  public static void main(String[] args) {
    try {
      System.out.println(Evaluation.evaluateModel(new Som(), args));
    }
    catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }


  public String mapWidthTipText() {
    return PARAM_DESCRIPTIONS[PARAM_MAP_WIDTH];
  }

  public String mapHeightTipText() {
    return PARAM_DESCRIPTIONS[PARAM_MAP_HEIGHT];
  }

  public String topologyTipText() {
    return PARAM_DESCRIPTIONS[PARAM_MAP_TOPOLOGY];
  }

  public String initialisationModeTipText() {
    return PARAM_DESCRIPTIONS[PARAM_MAP_INIT_MODE];
  }

  public String neighbourhoodFunctionTipText() {
    return PARAM_DESCRIPTIONS[PARAM_MAP_NEIGH_FUNC];
  }

  public String learningFunctionTipText() {
    return PARAM_DESCRIPTIONS[PARAM_MAP_LEARN_FUNC];
  }

  public String neighbourhoodSizeTipText() {
    return PARAM_DESCRIPTIONS[PARAM_MAP_NEIGH_SIZE];
  }

  public String trainingIterationsTipText() {
    return PARAM_DESCRIPTIONS[PARAM_MAP_TRAIN_ITER];
  }

  public String supervisedTipText() {
    return PARAM_DESCRIPTIONS[PARAM_MAP_SUPERVISED];
  }

  public String seedTipText() {
    return PARAM_DESCRIPTIONS[PARAM_MAP_SEED];
  }

  public String useVotingTipText() {
    return PARAM_DESCRIPTIONS[PARAM_MAP_VOTING];
  }


  /**
   * @return
   */
  public double getLearningRate() {
    return learningRate;
  }

  /**
   * @return
   */
  public int getMapHeight() {
    return mapHeight;
  }

  /**
   * @return
   */
  public int getMapWidth() {
    return mapWidth;
  }

  /**
   * @return
   */
  public int getNeighbourhoodSize() {
    return neighbourhoodSize;
  }

  /**
   * @return
   */
  public boolean isSupervised() {
    return supervised;
  }

  /**
   * @return
   */
  public int getTrainingIterations() {
    return trainingIterations;
  }

  /**
   * @param d
   */
  public void setLearningRate(double d) {
    learningRate = d;
  }

  /**
   * @param i
   */
  public void setMapHeight(int i) {
    mapHeight = i;
  }

  /**
   * @param i
   */
  public void setMapWidth(int i) {
    mapWidth = i;
  }

  /**
   * @param i
   */
  public void setNeighbourhoodSize(int i) {
    neighbourhoodSize = i;
  }

  /**
   * @param b
   */
  public void setSupervised(boolean b) {
    supervised = b;
  }

  /**
   * @param i
   */
  public void setTrainingIterations(int i) {
    trainingIterations = i;
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

  public void setTopology(SelectedTag s) {
    if (s.getTags() == NeighbourhoodDistanceFactory.TAGS_MODEL_TOPOLOGY) {
      mapTopology = s.getSelectedTag().getID();
    }
  }

  public SelectedTag getTopology() {
    return new SelectedTag(mapTopology, NeighbourhoodDistanceFactory.TAGS_MODEL_TOPOLOGY);
  }


  public void setNeighbourhoodFunction(SelectedTag s) {
    if (s.getTags() == NeighbourhoodKernelFactory.TAGS_NEIGHBOURHOOD_KERNEL) {
      neighbourhoodFunction = s.getSelectedTag().getID();
    }
  }

  public SelectedTag getNeighbourhoodFunction() {
    return new SelectedTag(neighbourhoodFunction, NeighbourhoodKernelFactory.TAGS_NEIGHBOURHOOD_KERNEL);
  }
}
