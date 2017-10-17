package weka.classifiers.neural.common;

import weka.core.Instance;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public interface BatchTrainableNeuralModel {

  public void calculateWeightErrors(Instance inputs, double aLearningRate);

  public void applyWeightDeltas(double aLearningRate);
}