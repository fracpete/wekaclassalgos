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

package weka.classifiers.neural.common.transfer;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public class SignTransferFunction extends TransferFunction {

  public final static double THRESHOLD = 0.0;

  public final static double MAX = +1.0;

  public final static double MIN = -1.0;


  public double transfer(double activation) {
    if (activation <= THRESHOLD) {
      return MIN;
    }

    // > 0.0
    return MAX;
  }

  public double derivative(double activation, double transferred) {
    //derivative is ( 2 * Dirac delta function )
    // yeah whatever, 2*0 or 0*INFINITY has no impact

    if (transferred == 0.0) {
      return Double.POSITIVE_INFINITY;
    }

    return 0.0;

  }

  public double getMaximum() {
    return MAX;
  }

  public double getMinimum() {
    return MIN;
  }
}