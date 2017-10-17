package weka.classifiers.neural.common;

import java.io.Serializable;
import java.util.Random;

/**
 * Date: 18/05/2004
 * File: RandomWrapper.java
 *
 * @author Jason Brownlee
 */
public class RandomWrapper implements Serializable {

  private long seed;

  private Random rand;

  /**
   * Constructor
   *
   * @param aSeed
   */
  public RandomWrapper(long aSeed) {
    seed = aSeed;
    rand = new Random(aSeed);
  }

  public void recreate() {
    rand = new Random(seed);
  }

  /**
   * Constructor
   */
  public RandomWrapper() {
    this(System.currentTimeMillis());
  }

  /**
   * @return
   */
  public Random getRand() {
    return rand;
  }

  /**
   * @return
   */
  public long getSeed() {
    return seed;
  }
}
