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


/**
 * Date: 25/05/2004
 * File: BubbleNeighbourhood.java
 *
 * @author Jason Brownlee
 */
public class BubbleNeighbourhood extends NeighbourhoodKernel {

  public BubbleNeighbourhood(double aInitialNeighborhood, int aTotalIterations) {
    super(aInitialNeighborhood, aTotalIterations);
  }

  public double calculateNeighbourhoodAdjustedLearningRate(double aCurrentLearningRate, double aDistance, double aCurrentNeighbourhoodSize) {
    return aCurrentLearningRate;
  }

  public boolean isDistanceInRadius(double aDistance, double aCurrentNeighbourhoodSize) {
    return (aDistance <= aCurrentNeighbourhoodSize);
  }

}
