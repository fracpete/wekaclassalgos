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

import weka.core.Attribute;
import weka.core.Instances;

/**
 * Description: Responsible for evaluating each attribute in the provided data instance,
 * and returning an ordered list of distance measures used to calculate the distance for each
 * attribute in the data item.
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class DistanceFactory {

  /**
   * Returns a vector of the same length as the data instance, where each element matches a
   * data attribute at the same index in the data instance, and provides
   * a suitable distance measure used for calculating the dinstanec between two values
   * of the attributes type. If the attribute is unknown, then a distance of OtherDistance
   * is assigned.
   *
   * @param instances
   * @return
   */
  public static AttributeDistance[] getAttributeDistanceList(Instances instances) {
    AttributeDistance[] distances = new AttributeDistance[instances.numAttributes()];

    for (int i = 0; i < distances.length; i++) {
      // check for class index
      if (i == instances.classIndex()) {
	distances[i] = new ClassDistance();
      }
      else {
	// select an appropriate distance metric
	switch (instances.attribute(i).type()) {
	  case Attribute.NOMINAL: {
	    distances[i] = new NominalDistance();
	    break;
	  }
	  case Attribute.DATE: // treat date as numeric (attribute does this)
	  case Attribute.NUMERIC: {
	    distances[i] = new NumericDistance();
	    break;
	  }
	  default: {
	    // support future types
	    distances[i] = new OtherDistance();
	    break;
	  }
	}
      }
    }

    return distances;
  }

  /**
   * Calculates the distances between the provided vectors, using the distance measures
   * provided.
   *
   * @param distances
   * @param instance
   * @param vector
   * @return
   */
  public final static double calculateDistance(AttributeDistance[] distances,
					       double[] instance,
					       double[] vector,
					       double aBestValue) {
    double result = 0.0;

    for (int i = 0; i < distances.length && result < aBestValue; i++) {
      // never compare a missing value to a valid value
      if (!weka.core.Utils.isMissingValue(instance[i])) {
	result += distances[i].distance(instance[i], vector[i]);
      }
    }

    return result;
  }
}