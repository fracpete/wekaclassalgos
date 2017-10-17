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

package weka.classifiers.neural.lvq.neighborhood;

import weka.core.Tag;

/**
 * Date: 25/05/2004
 * File: NeighbourhoodKernelFactory.java
 *
 * @author Jason Brownlee
 */
public class NeighbourhoodKernelFactory {

  public final static int NEIGHBOURHOOD_KERNEL_BUBBLE = 1;

  public final static int NEIGHBOURHOOD_KERNEL_GAUSSIAN = 2;

  // som neighbourhood kernel
  public final static Tag[] TAGS_NEIGHBOURHOOD_KERNEL =
    {
      new Tag(NEIGHBOURHOOD_KERNEL_BUBBLE, "Bubble"),
      new Tag(NEIGHBOURHOOD_KERNEL_GAUSSIAN, "Gaussian")
    };


  public final static String DESCRIPTION;

  static {
    StringBuffer buffer = new StringBuffer();
    buffer.append("(");

    for (int i = 0; i < TAGS_NEIGHBOURHOOD_KERNEL.length; i++) {
      buffer.append(TAGS_NEIGHBOURHOOD_KERNEL[i].getID());
      buffer.append("==");
      buffer.append(TAGS_NEIGHBOURHOOD_KERNEL[i].getReadable());

      if (i != TAGS_NEIGHBOURHOOD_KERNEL.length - 1) {
	buffer.append(", ");
      }
    }
    buffer.append(")");

    DESCRIPTION = buffer.toString();
  }

  public final static NeighbourhoodKernel factory(int aNeighbourhoodKernel, double aInitialNeighborhood, int aTotalIterations) {
    NeighbourhoodKernel kernel = null;

    switch (aNeighbourhoodKernel) {
      case NEIGHBOURHOOD_KERNEL_BUBBLE: {
	kernel = new BubbleNeighbourhood(aInitialNeighborhood, aTotalIterations);
	break;
      }
      case NEIGHBOURHOOD_KERNEL_GAUSSIAN: {
	kernel = new GaussianNeighbourhood(aInitialNeighborhood, aTotalIterations);
	break;
      }
      default: {
	throw new RuntimeException("Unknown neighbourhood kernel: " + aNeighbourhoodKernel);
      }
    }

    return kernel;
  }
}
