package weka.classifiers.neural.common.learning;

import java.io.Serializable;

/**
 * Date: 25/05/2004
 * File: LearningRateKernel.java
 *
 * @author Jason Brownlee
 */
public abstract class LearningRateKernel implements Serializable {

  protected final double initialLearningRate;

  protected final int totalIterations;

  public LearningRateKernel(double aLearningRate, int aTotalIterations) {
    initialLearningRate = aLearningRate;
    totalIterations = aTotalIterations;
  }


  public abstract double currentLearningRate(int aIteration);


  /**
   * @return
   */
  public double getInitialLearningRate() {
    return initialLearningRate;
  }

}
