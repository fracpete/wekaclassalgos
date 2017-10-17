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

package weka.classifiers.neural.lvq.initialise;

import weka.classifiers.neural.common.RandomWrapper;
import weka.core.Instances;
import weka.core.Tag;

/**
 * Date: 25/05/2004
 * File: InitialisationFactory.java
 *
 * @author Jason Brownlee
 */
public class InitialisationFactory {

  public final static int INITALISE_TRAINING_PROPORTIONAL = 1;

  public final static int INITALISE_TRAINING_EVEN = 2;

  public final static int INITALISE_RANDOM_VALUES = 3;

  public final static int INITALISE_SIMPLE_KMEANS = 4;

  public final static int INITALISE_FARTHEST_FIRST = 5;

  public final static int INITALISE_KNN = 6;


  public final static Tag[] TAGS_MODEL_INITALISATION =
    {
      new Tag(INITALISE_TRAINING_PROPORTIONAL, "Random Training Data Proportional"),
      new Tag(INITALISE_TRAINING_EVEN, "Random Training Data Even"),
      new Tag(INITALISE_RANDOM_VALUES, "Random Values In Range"),
      new Tag(INITALISE_SIMPLE_KMEANS, "Simple KMeans"),
      new Tag(INITALISE_FARTHEST_FIRST, "Farthest First"),
      new Tag(INITALISE_KNN, "K-Nearest Neighbour Even")
    };


  public final static String DESCRIPTION;

  static {
    StringBuffer buffer = new StringBuffer();
    buffer.append("(");

    for (int i = 0; i < TAGS_MODEL_INITALISATION.length; i++) {
      buffer.append(TAGS_MODEL_INITALISATION[i].getID());
      buffer.append("==");
      buffer.append(TAGS_MODEL_INITALISATION[i].getReadable());

      if (i != TAGS_MODEL_INITALISATION.length - 1) {
	buffer.append(", ");
      }
    }
    buffer.append(")");

    DESCRIPTION = buffer.toString();
  }


  public final static ModelInitialiser factory(int aInitialisationMode,
					       RandomWrapper aRand,
					       Instances aInstances,
					       int totalCodebookVectors) {
    ModelInitialiser initalise = null;

    switch (aInitialisationMode) {
      case INITALISE_TRAINING_PROPORTIONAL: {
	initalise = new RandomProportional(aRand, aInstances);
	break;
      }
      case INITALISE_TRAINING_EVEN: {
	initalise = new RandomEven(aRand, aInstances);
	break;
      }
      case INITALISE_RANDOM_VALUES: {
	initalise = new RandomValues(aRand, aInstances);
	break;
      }
      case INITALISE_SIMPLE_KMEANS: {
	initalise = new SimpleKMeansInitialiser(aRand, aInstances, totalCodebookVectors);
	break;
      }
      case INITALISE_FARTHEST_FIRST: {
	initalise = new FarthestFirstInitialiser(aRand, aInstances, totalCodebookVectors);
	break;
      }
      case INITALISE_KNN: {
	initalise = new KnnInitialiser(aRand, aInstances);
	break;
      }
      default: {
	throw new RuntimeException("Unknown initialisation mode: " + aInitialisationMode);
      }
    }

    return initalise;
  }
}
