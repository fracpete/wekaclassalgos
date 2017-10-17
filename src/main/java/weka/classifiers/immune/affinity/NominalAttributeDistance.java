/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.affinity;

/**
 * Type: NominalAttributeDistance
 * File: NominalAttributeDistance.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class NominalAttributeDistance implements AttributeDistance {

  /**
   * @param d1
   * @param d2
   * @return
   */
  public double distance(double d1, double d2) {
    if (d1 == d2) {
      return 0.0;
    }

    return 1.0;
  }


  public boolean isNumeric() {
    return false;
  }

  public boolean isClass() {
    return false;
  }

  public boolean isNominal() {
    return true;
  }

}
