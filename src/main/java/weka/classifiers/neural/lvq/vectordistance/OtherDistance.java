package weka.classifiers.neural.lvq.vectordistance;

/**
 * Description: Calculates the distance between two unknown attributes
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class OtherDistance implements AttributeDistance {

  /**
   * Distance between any attribute values, lower the distance the closer the values.
   * Always returns zero. Used for unknown attribute types
   *
   * @param instanceValue
   * @param codebookValue
   * @return
   */
  public double distance(double instanceValue, double codebookValue) {
    // not supported
    return 0.0;
  }
}