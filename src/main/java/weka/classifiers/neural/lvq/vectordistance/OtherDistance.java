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

package weka.classifiers.neural.lvq.vectordistance;

/**
 * Description: Calculates the distance between two unknown attributes
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class OtherDistance implements AttributeDistance {

  /**
   * Distance between any attribute values, lower the distance the closer the values.
   * Always returns zero. Used for unknown attribute types
   *
   * @param instanceValue
   * @param codebookValue
   * @return
   */
  public double distance(double instanceValue, double codebookValue) {
    // not supported
    return 0.0;
  }
}