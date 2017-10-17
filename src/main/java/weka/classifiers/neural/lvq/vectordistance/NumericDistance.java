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
 * Description: Distance between numberic attribute values, lower the
 * distance the closer the values.
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class NumericDistance implements AttributeDistance {

  /**
   * Distance between numberic attribute values, lower the distance the closer the values.
   * The square of the delta is returned. These can be summed to produce an approximation
   * of the eucliden distance (with or without the square root at the end).
   *
   * @param instanceValue
   * @param codebookValue
   * @return
   */
  public double distance(double instanceValue, double codebookValue) {
    // calculate the difference
    double delta = (instanceValue - codebookValue);
    // square the difference
    return (delta * delta);
  }
}