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

/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm;

import weka.classifiers.immune.affinity.AttributeDistance;
import weka.classifiers.immune.affinity.ClassAttributeDistance;
import weka.classifiers.immune.affinity.NominalAttributeDistance;
import weka.classifiers.immune.affinity.NumericAttributeDistance;
import weka.classifiers.immune.affinity.UnsupportedAttributeDistance;
import weka.core.Attribute;
import weka.core.Instances;

import java.io.Serializable;

/**
 * Type: AffinityFunction
 * File: AffinityFunction.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class DistanceFunction implements Serializable {

  protected final AttributeDistance[] distanceMeasures;

  protected final int classIndex;

  protected final double maxDistance;

  protected final double[][] minmax;


  public DistanceFunction(Instances aInstances) {
    int numAttributes = aInstances.numAttributes();
    distanceMeasures = new AttributeDistance[numAttributes];
    classIndex = aInstances.classIndex();
    // prepare distance measures
    prepareDistanceMeasures(aInstances);

    // calculate the maximum distance
    minmax = calculateMinMax(aInstances);

    // sum the squared ranges
    double sum = 0.0;
    for (int i = 0; i < aInstances.numAttributes(); i++) {
      Attribute a = aInstances.attribute(i);
      if (a.type() == Attribute.NUMERIC) {
	double range = (minmax[i][1] - minmax[i][0]);
	sum += (range * range);
      }
      else // non-numeric such as nominal
      {
	sum += 1.0; // 1 * 1
      }
    }

    maxDistance = Math.sqrt(sum);
  }

  protected double[][] calculateMinMax(Instances aInstances) {
    double[][] minmax = new double[aInstances.numAttributes()][2];

    for (int i = 0; i < minmax.length; i++) {
      minmax[i][0] = Double.POSITIVE_INFINITY;
      minmax[i][1] = Double.NEGATIVE_INFINITY;
    }

    for (int i = 0; i < aInstances.numInstances(); i++) {
      double[] attributes = aInstances.instance(i).toDoubleArray();
      for (int j = 0; j < attributes.length; j++) {
	// min
	if (attributes[j] < minmax[j][0]) {
	  minmax[j][0] = attributes[j];
	}
	// max
	if (attributes[j] > minmax[j][1]) {
	  minmax[j][1] = attributes[j];
	}
      }
    }

    return minmax;
  }

  public double[][] getMinMax() {
    return minmax;
  }

  public void prepareDistanceMeasures(Instances aInstances) {
    for (int i = 0; i < distanceMeasures.length; i++) {
      if (i == classIndex) {
	distanceMeasures[i] = new ClassAttributeDistance();
      }
      else {
	switch (aInstances.attribute(i).type()) {
	  case Attribute.NUMERIC: {
	    distanceMeasures[i] = new NumericAttributeDistance();
	    break;
	  }
	  case Attribute.NOMINAL: {
	    distanceMeasures[i] = new NominalAttributeDistance();
	    break;
	  }
	  // intentional fall through
	  case Attribute.DATE:
	  case Attribute.STRING: {
	    distanceMeasures[i] = new UnsupportedAttributeDistance();
	    break;
	  }
	  default: {
	    throw new RuntimeException("Unsupported attribute type: " + aInstances.attribute(i).type());
	  }
	}
      }
    }
  }

  public double distanceEuclideanNormalised(double[] i1, double[] i2) {
    double distance = calculateDistance(i1, i2);
    return normaliseDistance(distance);
  }

  public double distanceEuclideanUnnormalised(double[] i1, double[] i2) {
    return calculateDistance(i1, i2);
  }

  protected double calculateDistance(double[] i1, double[] i2) {
    // sum the squares
    double sum = 0.0;
    for (int i = 0; i < distanceMeasures.length; i++) {
      // check for empty value
      if (weka.core.Utils.isMissingValue(i1[i]) ||
	weka.core.Utils.isMissingValue(i2[i])) {
	// never attempt to compare missing values
	continue;
      }

      sum += distanceMeasures[i].distance(i1[i], i2[i]);
    }
    // square root
    double distance = Math.sqrt(sum);
    return distance;
  }

  protected double normaliseDistance(double aDistance) {
    return (aDistance / maxDistance);
  }
}
