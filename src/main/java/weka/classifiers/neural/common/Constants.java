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

package weka.classifiers.neural.common;

import weka.classifiers.neural.common.learning.LearningKernelFactory;
import weka.classifiers.neural.lvq.initialise.InitialisationFactory;


/**
 * Date: 25/05/2004
 * File: LvqConstants.java
 *
 * @author Jason Brownlee
 */
public interface Constants {

  // common paramter descriptions
  public final static String DESCRIPTION_CODEBOOK_VECTORS = "Total number of codebook vectors in the model";

  public final static String DESCRIPTION_TRAINING_ITERATIONS = "Total number of training iterations (recommended 30 to 50 times the number of codebook vectors).";

  public final static String DESCRIPTION_LEARNING_FUNCTION = "Learning rate function to use while training, linear is typically better " + LearningKernelFactory.DESCRIPTION;

  public final static String DESCRIPTION_LEARNING_RATE = "Initial learning rate value (recommend  0.3 or 0.5)";

  public final static String DESCRIPTION_WINDOW_SIZE = "Window size matching codebook vectors must be within (recommend 0.2 or 0.3)";

  public final static String DESCRIPTION_EPSILON = "Epsilon learning weight modifier used when both BMUs are of the instances class (recommend 0.1 or 0.5 should be smaller for smaller windowSize values).";

  public final static String DESCRIPTION_INITIALISATION = "Model (codebook vector) initalisation mode " + InitialisationFactory.DESCRIPTION;

  public final static String DESCRIPTION_RANDOM_SEED = "Random number generator seed, default 1, (whole numbers)";

  public final static String DESCRIPTION_USE_VOTING = "Use dynamic voting to select the assigned class of each codebook vector, provides automatic handling of misclassified instances.";


}