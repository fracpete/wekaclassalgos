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

package weka.classifiers.neural.lvq.topology;


/**
 * Date: 25/05/2004
 * File: RectangleNeighborhoodDistance.java
 *
 * @author Jason Brownlee
 */
public class RectangleNeighborhoodDistance implements NeighbourhoodDistance {


  public double neighborhoodDistance(int bx, int by, int tx, int ty) {
    double distance = 0.0;
    double sumSquares = 0.0;
    double diff = 0.0;

    // x
    diff = (double) bx - (double) tx;
    sumSquares += (diff * diff);
    // y
    diff = (double) by - (double) ty;
    sumSquares += (diff * diff);
    // square root the sum of the squared differences
    distance = Math.sqrt(sumSquares);
    return distance;
  }

}
