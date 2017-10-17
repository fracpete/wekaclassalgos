package weka.classifiers.neural.lvq.neighborhood;

import java.io.Serializable;


/**
 * Date: 25/05/2004
 * File: SomNeighborhood.java
 *
 * @author Jason Brownlee
 */
public abstract class NeighbourhoodKernel implements Serializable {

  protected final double initialNeighbourhoodSize;

  protected final int totalIterations;


  public NeighbourhoodKernel(double aInitialNeighborhood,
			     int aTotalIterations) {
    initialNeighbourhoodSize = aInitialNeighborhood;
    totalIterations = aTotalIterations;
  }

  public abstract boolean isDistanceInRadius(double aDistance, double aCurrentNeighbourhoodSize);

  public abstract double calculateNeighbourhoodAdjustedLearningRate(double aCurrentLearningRate, double aDistance, double aCurrentNeighbourhoodSize);


  public double currentNeighborhoodSize(int aCurrentIteration) {
    double currentRadius = 1.0 + (initialNeighbourhoodSize - 1.0) * (double) (totalIterations - aCurrentIteration) / (double) totalIterations;
    return currentRadius;
  }
}
