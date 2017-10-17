package weka.classifiers.neural.common.learning;

/**
 * Date: 25/05/2004
 * File: InverseLearningRate.java
 *
 * @author Jason Brownlee
 */
public class InverseLearningRate extends LearningRateKernel {

  private final static double INV_ALPHA_CONSTANT = 100.0;

  public InverseLearningRate(double aLearningRate, int aTotalIterations) {
    super(aLearningRate, aTotalIterations);
  }

  public double currentLearningRate(int aCurrentIteration) {
    double c = (double) totalIterations / INV_ALPHA_CONSTANT;
    double currentRate = (initialLearningRate * c / (c + (double) aCurrentIteration));
    return currentRate;
  }
}
