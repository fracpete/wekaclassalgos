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
 * File: HexagonalNeighbourhoodDistance.java
 *
 * @author Jason Brownlee
 */
public class HexagonalNeighbourhoodDistance implements NeighbourhoodDistance {

  public double neighborhoodDistance(int bx, int by, int tx, int ty) {
    double result = 0.0;
    double diff = 0.0;

    diff = (double) bx - (double) tx;

    // check for odd difference
    if (((by - ty) % 2) != 0) {
      // check for even row number
      if ((by % 2) == 0) {
	diff -= 0.5;
      }
      else {
	diff += 0.5;
      }
    }

    result = diff * diff;
    diff = (double) by - (double) ty;
    result += 0.75 * diff * diff;
    result = (double) Math.sqrt(result);

    return result;
  }
}
