package weka.classifiers.neural.lvq.topology;


/**
 * Date: 25/05/2004
 * File: RectangleNeighborhoodDistance.java
 * 
 * @author Jason Brownlee
 *
 */
public class RectangleNeighborhoodDistance implements NeighbourhoodDistance
{
	
	
	public double neighborhoodDistance(int bx, int by, int tx, int ty)
	{
		double distance = 0.0;
		double sumSquares = 0.0;
		double diff = 0.0;
		
		// x
		diff = (double)bx - (double)tx;
		sumSquares += (diff * diff);
		// y
		diff = (double)by - (double)ty;
		sumSquares += (diff * diff);
		// square root the sum of the squared differences
		distance = Math.sqrt(sumSquares);
		return distance;
	}

}
