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
package weka.classifiers.immune.affinity;

/**
 * Type: NumericAttributeDistance
 * File: NumericAttributeDistance.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class NumericAttributeDistance implements AttributeDistance {

  /**
   * @param d1
   * @param d2
   * @return
   */
  public double distance(double d1, double d2) {
    double diff = (d1 - d2);
    return (diff * diff);
  }


  public boolean isNumeric() {
    return true;
  }

  public boolean isClass() {
    return false;
  }

  public boolean isNominal() {
    return false;
  }
}
