package weka.classifiers.neural.lvq.topology;

import java.io.Serializable;

/**
 * Date: 25/05/2004
 * File: NeighbourhoodDistance.java
 * 
 * @author Jason Brownlee
 *
 */
public interface NeighbourhoodDistance extends Serializable
{
	double neighborhoodDistance(int bx, int by, int tx, int ty);
}
