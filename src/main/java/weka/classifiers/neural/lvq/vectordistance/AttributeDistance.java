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

import java.io.Serializable;

/**
 * Description: Common interface used to calculate an attributes distance. The smaller
 * the distance value, the closer the values match
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public interface AttributeDistance extends Serializable {

  /**
   * Distance between attribute values, lower the distnace the closer the values
   *
   * @param instanceValue
   * @param codebookValue
   * @return
   */
  double distance(double instanceValue, double codebookValue);
}