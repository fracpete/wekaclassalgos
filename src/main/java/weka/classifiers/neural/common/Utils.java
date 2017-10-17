package weka.classifiers.neural.common;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public class Utils {

  public static double max(double[] vector) {
    double max = vector[0];
    for (int i = 1; i < vector.length; i++) {
      if (vector[i] > max) {
	max = vector[i];
      }
    }

    return max;
  }

  public static double min(double[] vector) {
    double min = vector[0];
    for (int i = 1; i < vector.length; i++) {
      if (vector[i] < min) {
	min = vector[i];
      }
    }

    return min;
  }

  // normalise the provided vector
  public static void normalise(double[] vector) {
    double max = max(vector);
    double min = min(vector);
    normalise(vector, max, min);
  }

  public static void normalise(double[] vector, double max, double min) {
    double range = (max - min);

    for (int i = 0; i < vector.length; i++) {
      vector[i] = ((vector[i] - min) / range);
    }
  }

}