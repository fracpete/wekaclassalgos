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

public class HyperbolicTangentTransferFunction extends TransferFunction {

  public final static double MAX = +1.0;

  public final static double MIN = -1.0;


  public double transfer(double activation) {
    // poor implementation that breaks when saturated with values > +-709
    //return ( (Math.exp(activation) - Math.exp(-activation)) / (Math.exp(activation) + Math.exp(-activation)) );

    // more accurate implementation that returns true values when saturated
    // y = 2 / (1 + exp(-2 * x)) - 1
    return 2.0 / (1.0 + Math.exp(-2.0 * activation)) - 1.0;
  }

  public double derivative(double activation, double transferred) {
    // 1 - y * y
    return (1.0 - (transferred * transferred));
  }

  public double getMaximum() {
    return MAX;
  }

  public double getMinimum() {
    return MIN;
  }
}