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

public class GaussianTransferFunction extends TransferFunction {

  public final static double MAX = +1.0;

  public final static double MIN = 0.0;


  public double transfer(double activation) {
    // y = exp(- x * x)
    return Math.exp(-activation * activation);
  }

  public double derivative(double activation, double transferred) {
    // -2 * sum * output
    return -2.0 * activation * transferred;
  }


  public double getMaximum() {
    return MAX;
  }

  public double getMinimum() {
    return MIN;
  }
}