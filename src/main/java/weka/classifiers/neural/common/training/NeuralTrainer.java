package weka.classifiers.neural.common.training;

import weka.classifiers.neural.common.NeuralModel;
import weka.classifiers.neural.common.RandomWrapper;
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

public abstract class NeuralTrainer implements Serializable {

  protected final RandomWrapper rand;

  public NeuralTrainer(RandomWrapper aRand) {
    rand = aRand;
  }

  public abstract void trainModel(NeuralModel model, Instances instances, int numIterations);
}