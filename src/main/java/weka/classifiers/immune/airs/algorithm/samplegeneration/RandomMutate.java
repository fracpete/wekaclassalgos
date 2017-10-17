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
