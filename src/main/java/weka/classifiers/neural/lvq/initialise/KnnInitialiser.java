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

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.neural.common.RandomWrapper;
import weka.core.Instance;
import weka.core.Instances;

import java.util.LinkedList;

/**
 * Date: 26/05/2004
 * File: KnnInitialiser.java
 *
 * @author Jason Brownlee
 */
public class KnnInitialiser extends CommonInitialiser {

  protected Classifier classifier;

  protected int roundRobbinIndex;

  private boolean fullCircle;

  protected final LinkedList[] trainingDataClassDistribution;


  public KnnInitialiser(RandomWrapper aRand, Instances aInstances) {
    super(aRand, aInstances);
    trainingDataClassDistribution = new LinkedList[numClasses];
    prepareClassifier();
  }

  protected Classifier getClassifier() {
    IBk algorithm = new IBk();
    algorithm.setKNN(7);
    return algorithm;
  }

  protected void prepareClassifier() {
    // construct the classifier
    classifier = getClassifier();
    try {
      // train the classifier
      classifier.buildClassifier(trainingInstances);

      // collect all instances correctly classified
      for (int i = 0; i < trainingInstances.numInstances(); i++) {
	Instance instance = trainingInstances.instance(i);
	double classification = classifier.classifyInstance(instance);
	// only collect those instances that are classified correctly
	if (classification == instance.classValue()) {
	  int index = (int) classification;
	  if (trainingDataClassDistribution[index] == null) {
	    trainingDataClassDistribution[index] = new LinkedList();
	  }

	  trainingDataClassDistribution[index].add(instance);
	}
      }
    }
    catch (Exception e) {
      throw new RuntimeException("Failed to prepare classifier: " + e.getMessage(), e);
    }
  }

  public double[] getAttributes() {
    // select an instance
    Instance instance = selectInstance();
    double[] attributes = instance.toDoubleArray();
    return attributes;
  }


  protected Instance selectInstance() {
    Instance selectedInstance = null;
    int startRoundRobbinValue = roundRobbinIndex;

    while (selectedInstance == null && !fullCircle) {
      // check for an empty class on the current round robbin
      if (trainingDataClassDistribution[roundRobbinIndex] == null) {
	incrementRoundRobbin();
	// check for full circle
	if (roundRobbinIndex == startRoundRobbinValue) {
	  fullCircle = true;
	}
      }
      // check if all vectors from the current class have been included in the model
      else if (trainingDataClassDistribution[roundRobbinIndex].isEmpty()) {
	incrementRoundRobbin();
	// check for full circle
	if (roundRobbinIndex == startRoundRobbinValue) {
	  fullCircle = true;
	}
      }
      // the current selection can be used
      else {
	int selection = makeRandomSelection(trainingDataClassDistribution[roundRobbinIndex].size());
	selectedInstance = (Instance) trainingDataClassDistribution[roundRobbinIndex].remove(selection);
	incrementRoundRobbin();
      }
    }

    if (fullCircle) {
      // select any random instance
      int selection = makeRandomSelection(totalInstances);
      selectedInstance = trainingInstances.instance(selection);
    }

    return selectedInstance;
  }

  protected void incrementRoundRobbin() {
    if (++roundRobbinIndex >= numClasses) {
      roundRobbinIndex = 0;
    }
  }
}
