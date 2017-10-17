package weka.classifiers.neural.lvq.vectordistance;


/**
 * Description: Distance between numberic attribute values, lower the
 * distance the closer the values.
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class NumericDistance implements AttributeDistance {

  /**
   * Distance between numberic attribute values, lower the distance the closer the values.
   * The square of the delta is returned. These can be summed to produce an approximation
   * of the eucliden distance (with or without the square root at the end).
   *
   * @param instanceValue
   * @param codebookValue
   * @return
   */
  public double distance(double instanceValue, double codebookValue) {
    // calculate the difference
    double delta = (instanceValue - codebookValue);
    // square the difference
    return (delta * delta);
  }
}