package weka.classifiers.neural.common;

import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public interface NeuralModel extends Serializable {

  // update the model for a single pattern
  public void updateModel(Instance instance, double aLearningRate);

  public void startingEpoch();

  public void finishedEpoch(Instances instances, double aLearningRate);

  public double getLearningRate(int epochNumber);


  // run a vector through the model and get a result
  public double[] getNetworkOutputs(Instance instance);

  public double[] getDistributionForInstance(Instance instance);

  // stats
  public String getModelInformation();

  public int getNumOutputNeurons();

  public double[] getAllWeights();
}