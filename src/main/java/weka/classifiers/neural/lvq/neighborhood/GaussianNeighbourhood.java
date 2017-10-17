package weka.classifiers.neural.lvq.neighborhood;


/**
 * Date: 25/05/2004
 * File: GaussianNeighbourhood.java
 * 
 * @author Jason Brownlee
 *
 */
public class GaussianNeighbourhood extends NeighbourhoodKernel
{
	public GaussianNeighbourhood(double aInitialNeighborhood, int aTotalIterations)
	{
		super(aInitialNeighborhood, aTotalIterations);
	}

	public double calculateNeighbourhoodAdjustedLearningRate(double aCurrentLearningRate, double aDistance, double aCurrentNeighbourhoodSize)
	{		
		return aCurrentLearningRate * Math.exp((-aDistance * aDistance / (2.0 * aCurrentNeighbourhoodSize * aCurrentNeighbourhoodSize)));
	}

	public boolean isDistanceInRadius(double aDistance, double aCurrentNeighbourhoodSize)
	{
		return true;
	}
}
