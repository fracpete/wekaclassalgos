package weka.classifiers.neural.common.learning;

/**
 * Date: 25/05/2004
 * File: LinearLearningRate.java
 *
 * @author Jason Brownlee
 */
public class LinearLearningRate extends LearningRateKernel {

  public LinearLearningRate(double aLearningRate, int aTotalIterations) {
    super(aLearningRate, aTotalIterations);
  }

  public double currentLearningRate(int aCurrentIteration) {
    double currentRate = (initialLearningRate * (double) (totalIterations - aCurrentIteration) / (double) totalIterations);
    return currentRate;
  }
}
