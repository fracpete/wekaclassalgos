package weka.classifiers.neural.lvq.neighborhood;


/**
 * Date: 25/05/2004
 * File: BubbleNeighbourhood.java
 * 
 * @author Jason Brownlee
 *
 */
public class BubbleNeighbourhood extends NeighbourhoodKernel
{
	public BubbleNeighbourhood(double aInitialNeighborhood, int aTotalIterations)
	{
		super(aInitialNeighborhood, aTotalIterations);
	}

	public double calculateNeighbourhoodAdjustedLearningRate(double aCurrentLearningRate, double aDistance, double aCurrentNeighbourhoodSize)
	{		
		return aCurrentLearningRate;
	}

	public boolean isDistanceInRadius(double aDistance, double aCurrentNeighbourhoodSize)
	{
		return (aDistance <= aCurrentNeighbourhoodSize);
	}

}
