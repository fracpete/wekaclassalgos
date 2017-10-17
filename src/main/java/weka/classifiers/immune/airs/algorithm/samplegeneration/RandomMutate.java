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
 */
package weka.classifiers.immune.airs.algorithm.samplegeneration;

import weka.classifiers.immune.airs.algorithm.Cell;
import weka.classifiers.immune.airs.algorithm.SampleGenerator;
import weka.core.Attribute;
import weka.core.Instance;

import java.util.Random;

/**
 * Type: RandomMutate File: RandomMutate.java Date: 30/12/2004 Description:
 *
 * @author Jason Brownlee
 */
public class RandomMutate extends SampleGenerator {

  protected final int numClasses;

  protected final double mutationRate;

  public RandomMutate(
    Random aRand,
    int aNumClasses,
    double aMutationRate) {
    super(aRand);
    numClasses = aNumClasses;
    mutationRate = aMutationRate;
  }


  public Cell generateSample(Cell aCell, Instance aInstance) {
    Cell c = new Cell(aCell);
    double[] attributes = c.getAttributes();
    boolean didMutate = false;

    // continue until one mutation occurs
    do {
      for (int i = 0; i < attributes.length; i++) {
	int type = aInstance.attribute(i).type();

	double canMutate = rand.nextDouble();
	if (canMutate < mutationRate) {
	  if (type == Attribute.NUMERIC) {
	    // random value
	    attributes[i] = rand.nextDouble();
	    didMutate = true;
	  }
	  else if (type == Attribute.NOMINAL) {
	    // random nominal value
	    attributes[i] = rand.nextInt(aInstance.attribute(i).numValues());
	    didMutate = true;
	  }
	}
      }
    }
    while (!didMutate);

    return c;
  }
}
