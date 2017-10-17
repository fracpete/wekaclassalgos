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
 * Description: Calculates the distance between two class attributes.
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class ClassDistance implements AttributeDistance {

  /**
   * Distance between class attribute values, lower the distnace the closer the values
   *
   * @param instanceValue
   * @param codebookValue
   * @return
   */
  public double distance(double instanceValue, double codebookValue) {
    // never calculate a distance for the class value
    return 0.0;
  }
}