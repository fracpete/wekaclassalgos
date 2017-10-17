/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
