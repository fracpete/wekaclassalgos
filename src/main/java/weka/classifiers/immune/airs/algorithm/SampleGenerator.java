/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm;

import weka.core.Instance;

import java.util.Random;

/**
 * Type: CloneGenerator
 * File: CloneGenerator.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public abstract class SampleGenerator {

  protected final Random rand;

  public SampleGenerator(Random aRand) {
    rand = aRand;
  }


  public abstract Cell generateSample(Cell aCell, Instance aInstance);
}
