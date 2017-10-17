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

import weka.core.Instance;
import weka.core.Instances;

/**
 * Type: AffinityFunction
 * File: AffinityFunction.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class AffinityFunction extends DistanceFunction {

  public AffinityFunction(Instances aInstances) {
    super(aInstances);
  }

  public double affinityNormalised(double[] i1, double[] i2) {
    // single point for adjustment
    return distanceEuclideanNormalised(i1, i2);
  }

  public double affinityUnnormalised(double[] i1, double[] i2) {
    // single point for adjustment
    return distanceEuclideanUnnormalised(i1, i2);
  }


  public double affinityNormalised(Instance i1, Instance i2) {
    return affinityNormalised(i1.toDoubleArray(), i2.toDoubleArray());
  }

  public double affinityNormalised(Instance i1, Cell c2) {
    return affinityNormalised(i1.toDoubleArray(), c2.getAttributes());
  }

  public double affinityNormalised(double[] i1, Cell c2) {
    return affinityNormalised(i1, c2.getAttributes());
  }

  public double affinityNormalised(Cell c1, Cell c2) {
    return affinityNormalised(c1.getAttributes(), c2.getAttributes());
  }


  public double affinityUnnormalised(Instance i1, Instance i2) {
    return affinityUnnormalised(i1.toDoubleArray(), i2.toDoubleArray());
  }

  public double affinityUnnormalised(Instance i1, Cell c2) {
    return affinityUnnormalised(i1.toDoubleArray(), c2.getAttributes());
  }

  public double affinityUnnormalised(double[] i1, Cell c2) {
    return affinityUnnormalised(i1, c2.getAttributes());
  }

  public double affinityUnnormalised(Cell c1, Cell c2) {
    return affinityUnnormalised(c1.getAttributes(), c2.getAttributes());
  }

}
