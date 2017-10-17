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

package weka.classifiers.neural.lvq.model;

import weka.classifiers.neural.lvq.topology.NeighbourhoodDistance;

/**
 * Date: 25/05/2004
 * File: SOMModel.java
 *
 * @author Jason Brownlee
 */
public class SomModel extends CommonModel {

  protected final NeighbourhoodDistance neighbourhoodDistance;

  protected final int mapWidth;

  protected final int mapHeight;


  public SomModel(NeighbourhoodDistance aNeighbourhoodDistance,
		  int aMapWidth,
		  int aMapHeight) {
    super(aMapWidth * aMapHeight);
    neighbourhoodDistance = aNeighbourhoodDistance;
    mapWidth = aMapWidth;
    mapHeight = aMapHeight;
  }


  public double calculateNeighbourhoodDistance(CodebookVector aBmu, CodebookVector aVector) {
    // determine vector rectangular coordinates
    int bx = aBmu.getId() % mapWidth;
    int by = aBmu.getId() / mapWidth;
    int tx = aVector.getId() % mapWidth;
    int ty = aVector.getId() / mapWidth;
    // calculate distance
    return neighbourhoodDistance.neighborhoodDistance(bx, by, tx, ty);
  }
}
