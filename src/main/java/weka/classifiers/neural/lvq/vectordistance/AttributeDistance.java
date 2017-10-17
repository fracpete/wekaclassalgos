package weka.classifiers.neural.lvq.vectordistance;

import java.io.Serializable;

/**
 * Description: Common interface used to calculate an attributes distance. The smaller
 * the distance value, the closer the values match
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public interface AttributeDistance extends Serializable {

  /**
   * Distance between attribute values, lower the distnace the closer the values
   *
   * @param instanceValue
   * @param codebookValue
   * @return
   */
  double distance(double instanceValue, double codebookValue);
}