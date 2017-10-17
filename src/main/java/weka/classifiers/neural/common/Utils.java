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

package weka.classifiers.neural.common;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public class Utils {

  public static double max(double[] vector) {
    double max = vector[0];
    for (int i = 1; i < vector.length; i++) {
      if (vector[i] > max) {
	max = vector[i];
      }
    }

    return max;
  }

  public static double min(double[] vector) {
    double min = vector[0];
    for (int i = 1; i < vector.length; i++) {
      if (vector[i] < min) {
	min = vector[i];
      }
    }

    return min;
  }

  // normalise the provided vector
  public static void normalise(double[] vector) {
    double max = max(vector);
    double min = min(vector);
    normalise(vector, max, min);
  }

  public static void normalise(double[] vector, double max, double min) {
    double range = (max - min);

    for (int i = 0; i < vector.length; i++) {
      vector[i] = ((vector[i] - min) / range);
    }
  }

}