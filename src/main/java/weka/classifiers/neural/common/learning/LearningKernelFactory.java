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

package weka.classifiers.neural.common.learning;

import weka.core.Tag;

/**
 * Date: 25/05/2004
 * File: LearningRateFactory.java
 *
 * @author Jason Brownlee
 */
public class LearningKernelFactory {

  public final static int LEARNING_FUNCTION_LINEAR = 1;

  public final static int LEARNING_FUNCTION_INVERSE = 2;

  public final static int LEARNING_FUNCTION_STATIC = 3;

  public final static Tag[] TAGS_LEARNING_FUNCTION =
    {
      new Tag(LEARNING_FUNCTION_LINEAR, "Linear Decay"),
      new Tag(LEARNING_FUNCTION_INVERSE, "Inverse"),
      new Tag(LEARNING_FUNCTION_STATIC, "Static")
    };

  public final static String[] LEARNING_FUNCTION_FULL_DESC =
    {
      "Linear decay learning rate function",
      "Inverse learning rate function",
      "Static learning rate"
    };

  public static String getDescription(int aLearningFunction) {
    return LEARNING_FUNCTION_FULL_DESC[aLearningFunction - 1];
  }

  public final static String DESCRIPTION;

  static {
    StringBuilder buffer = new StringBuilder();
    buffer.append("(");

    for (int i = 0; i < TAGS_LEARNING_FUNCTION.length; i++) {
      buffer.append(TAGS_LEARNING_FUNCTION[i].getID());
      buffer.append("==");
      buffer.append(TAGS_LEARNING_FUNCTION[i].getReadable());

      if (i != TAGS_LEARNING_FUNCTION.length - 1) {
	buffer.append(", ");
      }
    }
    buffer.append(")");

    DESCRIPTION = buffer.toString();
  }

  public final static LearningRateKernel factory(int aLearningRate, double initalLearningRate, int totalIterations) {
    LearningRateKernel kernel = null;

    switch (aLearningRate) {
      case LEARNING_FUNCTION_LINEAR: {
	kernel = new LinearLearningRate(initalLearningRate, totalIterations);
	break;
      }
      case LEARNING_FUNCTION_INVERSE: {
	kernel = new InverseLearningRate(initalLearningRate, totalIterations);
	break;
      }
      case LEARNING_FUNCTION_STATIC: {
	kernel = new StaticLearningRate(initalLearningRate, totalIterations);
	break;
      }
      default: {
	throw new RuntimeException("Unknown learning rate: " + aLearningRate);
      }
    }

    return kernel;
  }
}
