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

package weka.classifiers.neural.common.initialisation;

import weka.classifiers.neural.common.RandomWrapper;


/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public class Initialisation {

  public static void initialiseVectorToRandom(double[] vector,
					      double upper,
					      double lower,
					      RandomWrapper rand) {
    for (int i = 0; i < vector.length; i++) {
      vector[i] = getRandomDouble(upper, lower, rand);
    }
  }

  public static void initialiseVectorToRandomWithSign(double[] vector,
						      double upper,
						      double lower,
						      RandomWrapper rand) {
    for (int i = 0; i < vector.length; i++) {
      vector[i] = getRandomDoubleWithSign(upper, lower, rand);
    }
  }


  // Generate random double between the two specified ranges.
  // Ranges limited to 0.0 amd 1.0
  public static double getRandomDouble(double upperRange,
				       double lowerRange,
				       RandomWrapper rand) {
    return lowerRange + (rand.getRand().nextDouble() * (upperRange - lowerRange));
  }


  // Generate a random double between the two specified ranges.
  // Ranges are limited to 0.0 and 1.0, the number produced will be randomly
  // either positive or negative (between -1.0 and +1.0)
  public static double getRandomDoubleWithSign(double upperRange,
					       double lowerRange,
					       RandomWrapper rand) {
    double value = getRandomDouble(lowerRange, upperRange, rand);

    // 50% chance of a negative
    if (rand.getRand().nextBoolean()) {
      return -value;
    }

    return value;
  }

}