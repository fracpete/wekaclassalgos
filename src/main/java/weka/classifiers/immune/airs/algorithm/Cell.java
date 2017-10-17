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

/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm;

import weka.core.Instance;

import java.io.Serializable;

/**
 * Type: Cell
 * File: Cell.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class Cell implements Serializable {

  private final double[] attributes;

  private final int classIndex;

  private long usage;

  /**
   *
   */
  protected double affinity;

  /**
   * number of resources held by the cell
   */
  protected double numResources;

  /**
   * current stimulation value
   */
  protected double stimulation;


  public Cell(double[] aAttributes, int aClassIndex) {
    attributes = aAttributes;
    classIndex = aClassIndex;
  }

  public Cell(Instance aInstance) {
    // note to double array creates a new object
    this(aInstance.toDoubleArray(), aInstance.classIndex());
  }


  public Cell(Cell aCell) {
    classIndex = aCell.classIndex;
    attributes = new double[aCell.attributes.length];
    System.arraycopy(aCell.attributes, 0, attributes, 0, attributes.length);
  }


  public double getClassification() {
    return attributes[classIndex];
  }

  public double[] getAttributes() {
    return attributes;
  }

  public int getClassIndex() {
    return classIndex;
  }


  public double getAffinity() {
    return affinity;
  }

  public void setAffinity(double affinity) {
    this.affinity = affinity;
  }


  public double getNumResources() {
    return numResources;
  }

  public void setNumResources(double numResources) {
    this.numResources = numResources;
  }

  public double getStimulation() {
    return stimulation;
  }

  public void setStimulation(double stimulation) {
    this.stimulation = stimulation;
  }

  protected long getUsage() {
    return usage;
  }

  protected void incrementUsage() {
    usage++;
  }

  protected void clearUsage() {
    usage = 0;
  }
}
