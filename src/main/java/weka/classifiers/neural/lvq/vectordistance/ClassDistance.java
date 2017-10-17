package weka.classifiers.neural.lvq.vectordistance;

/**
 * Description: Calculates the distance between two class attributes.
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class ClassDistance implements AttributeDistance {

  /**
   * Distance between class attribute values, lower the distnace the closer the values
   *
   * @param instanceValue
   * @param codebookValue
   * @return
   */
  public double distance(double instanceValue, double codebookValue) {
    // never calculate a distance for the class value
    return 0.0;
  }
}