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

import java.io.Serializable;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public abstract class TransferFunction implements Serializable {

  public final static double UPPER_THREAHOLD = +45.0;

  public final static double LOWER_THREAHOLD = -45.0;


  // from the NN FAQ on overflow protection
  public double overflowProtectionTransfer(double activation) {
    double output = 0.0;

    if (activation < LOWER_THREAHOLD) {
      output = getMinimum();
    }
    else if (activation > UPPER_THREAHOLD) {
      output = getMaximum();
    }
    else {
      output = transfer(activation);
    }

    return output;
  }

  public abstract double transfer(double activation);

  public abstract double derivative(double activation, double transferted);

  public abstract double getMaximum();

  public abstract double getMinimum();
}