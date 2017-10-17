/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.affinity;


/**
 * Type: UnsupportedAttributeDistance
 * File: UnsupportedAttributeDistance.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class UnsupportedAttributeDistance implements AttributeDistance {

  /**
   * @param d1
   * @param d2
   * @return
   */
  public double distance(double d1, double d2) {
    throw new RuntimeException("The data instance contains an unsupported attribute type for the vector distance measure.");
  }


  public boolean isNumeric() {
    return false;
  }

  public boolean isClass() {
    return false;
  }

  public boolean isNominal() {
    return false;
  }
}
