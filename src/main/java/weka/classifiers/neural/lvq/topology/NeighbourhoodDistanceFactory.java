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

import weka.core.Tag;

/**
 * Date: 25/05/2004
 * File: NeighbourhoodDistanceFactory.java
 *
 * @author Jason Brownlee
 */
public class NeighbourhoodDistanceFactory {

  public final static int NEIGHBOURHOOD_DISTANCE_RECTANGLE = 1;

  public final static int NEIGHBOURHOOD_DISTNACE_HEXAGONAL = 2;

  public final static Tag[] TAGS_MODEL_TOPOLOGY =
    {
      new Tag(NEIGHBOURHOOD_DISTANCE_RECTANGLE, "Rectangular"),
      new Tag(NEIGHBOURHOOD_DISTNACE_HEXAGONAL, "Hexagonal")
    };

  public final static String DESCRIPTION;

  static {
    StringBuilder buffer = new StringBuilder();
    buffer.append("(");

    for (int i = 0; i < TAGS_MODEL_TOPOLOGY.length; i++) {
      buffer.append(TAGS_MODEL_TOPOLOGY[i].getID());
      buffer.append("==");
      buffer.append(TAGS_MODEL_TOPOLOGY[i].getReadable());

      if (i != TAGS_MODEL_TOPOLOGY.length - 1) {
	buffer.append(", ");
      }
    }
    buffer.append(")");

    DESCRIPTION = buffer.toString();
  }

  public final static NeighbourhoodDistance factory(int aNeighbourhoodDistance) {
    NeighbourhoodDistance distance = null;

    switch (aNeighbourhoodDistance) {
      case NEIGHBOURHOOD_DISTANCE_RECTANGLE: {
	distance = new RectangleNeighborhoodDistance();
	break;
      }
      case NEIGHBOURHOOD_DISTNACE_HEXAGONAL: {
	distance = new HexagonalNeighbourhoodDistance();
	break;
      }
      default: {
	throw new RuntimeException("Unknown neighbourhood distance: " + aNeighbourhoodDistance);
      }
    }

    return distance;
  }
}
