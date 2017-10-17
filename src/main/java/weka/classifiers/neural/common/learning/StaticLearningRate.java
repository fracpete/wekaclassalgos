package weka.classifiers.neural.common.learning;

/**
 * Date: 25/05/2004
 * File: StaticLearningRate.java
 *
 * @author Jason Brownlee
 */
public class StaticLearningRate extends LearningRateKernel {

  public StaticLearningRate(double aLearningRate, int aTotalIterations) {
    super(aLearningRate, aTotalIterations);
  }

  public double currentLearningRate(int aIteration) {
    return initialLearningRate;
  }
}
