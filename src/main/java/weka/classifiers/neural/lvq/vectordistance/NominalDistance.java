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
 * Description: Calculates the distance between two nominal data values
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class NominalDistance implements AttributeDistance {

  /**
   * Distance between nominal attribute values, lower the distnace the closer the values.
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

    //
    // JB 24May2004
    // Note: I don't like this idea of binary comparison - the return value
    // assumes too much about the data, who's to know if 1.0 is meaningful or too meaningful
    //
    // binary comparison
    //return (instanceValue == codebookValue) ? 0.0 : 1.0;
  }
}