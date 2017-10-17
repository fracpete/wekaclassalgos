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
 * Created on 23/01/2005
 *
 */
package weka.classifiers.immune.immunos;

import weka.core.Instance;

/**
 * Type: Immunos99Antibody<br>
 * File: Immunos99Antibody.java<br>
 * Date: 23/01/2005<br>
 * <br>
 * Description:
 * <br>
 *
 * @author Jason Brownlee
 */
public class Immunos99Antibody extends Antibody {

  protected final int numClasses;

  protected final double[] classCounts;

  protected double fitness;


  public Immunos99Antibody(
    double[] aAttributes,
    int aClassIndex,
    int aNumClasses) {
    super(aAttributes, aClassIndex);
    numClasses = aNumClasses;
    classCounts = new double[numClasses];
  }

  public Immunos99Antibody(Instance aInstance) {
    super(aInstance);
    numClasses = aInstance.classAttribute().numValues();
    classCounts = new double[numClasses];
  }

  public Immunos99Antibody(Immunos99Antibody aParent) {
    super(aParent);
    numClasses = aParent.numClasses;
    classCounts = new double[numClasses];
  }


  public void updateClassCount(Instance aInstance, double score) {
    classCounts[(int) aInstance.classValue()] += score;
  }

  public void clearClassCounts() {
    for (int i = 0; i < classCounts.length; i++) {
      classCounts[i] = 0.0;
    }
  }

  public boolean hasMisClassified() {
    for (int i = 0; i < classCounts.length; i++) {
      if (i != (int) getClassification() && classCounts[i] > 0) {
	return true;
      }
    }

    return false;
  }

  public boolean canSwitchClass() {
    // no correct
    if (classCounts[(int) getClassification()] == 0) {
      // has some missing
      if (hasMisClassified()) {
	return true;
      }
    }

    // have some instances
    return false;
  }

  public void switchClasses() {
    double best = -1;
    int bestIndex = -1;

    for (int i = 0; i < classCounts.length; i++) {
      if (classCounts[i] > best) {
	best = classCounts[i];
	bestIndex = i;
      }
    }

    // assign new class
    attributes[classIndex] = bestIndex;
  }

  public double calculateFitness() {
    double totalCorrect = classCounts[(int) getClassification()];
    double totalIncorrect = 0.0;
    for (int i = 0; i < classCounts.length; i++) {
      if (i != (int) getClassification()) {
	totalIncorrect += classCounts[i];
      }
    }

    if (totalCorrect == 0) {
      // have nothing correct
      fitness = 0.0;
    }
    else if (totalIncorrect == 0) {
      // have some correct, and no incorrect
      fitness = totalCorrect;
    }
    else {
      fitness = (totalCorrect / totalIncorrect);
    }

    return fitness;
  }

  public double getFitness() {
    return fitness;
  }
}
