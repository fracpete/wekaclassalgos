/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
