package weka.classifiers.immune.clonalg;

import weka.core.Instance;

import java.io.Serializable;

/**
 * Type: Antibody<br>
 * Date: 19/01/2005<br>
 * <br>
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class Antibody implements Comparable<Antibody>, Serializable {

  protected final double[] attributes;

  protected final int classIndex;

  protected double affinity;


  public Antibody(double[] aAttributes, int aClassIndex) {
    attributes = aAttributes;
    classIndex = aClassIndex;
  }

  public Antibody(Instance aInstance) {
    this(aInstance.toDoubleArray(), aInstance.classIndex());
  }

  public Antibody(Antibody aParent) {
    double[] copy = new double[aParent.attributes.length];
    System.arraycopy(aParent.attributes, 0, copy, 0, copy.length);
    attributes = copy;
    classIndex = aParent.classIndex;
  }


  public double getClassification() {
    return attributes[classIndex];
  }

  /**
   * Compares this object with the specified object for order.  Returns a
   * negative integer, zero, or a positive integer as this object is less
   * than, equal to, or greater than the specified object.<p>
   * <p>
   * Ascending order meaning smallest value to the largest value. This is infact
   * decending order in regard to affinity quality, as the lower the value
   * the higher the affinity
   */
  public int compareTo(Antibody other) {
    if (affinity < other.affinity) {
      return -1;
    }
    else if (affinity > other.affinity) {
      return +1;
    }

    return 0;
  }

  /**
   * @return Returns the attributes.
   */
  public double[] getAttributes() {
    return attributes;
  }

  /**
   * @return Returns the classIndex.
   */
  public int getClassIndex() {
    return classIndex;
  }


  /**
   * @return Returns the affinity.
   */
  public double getAffinity() {
    return affinity;
  }

  /**
   * @param affinity The affinity to set.
   */
  public void setAffinity(double affinity) {
    this.affinity = affinity;
  }
}
