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
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.Serializable;
import java.util.LinkedList;

/**
 * Type: AISModelClassifier
 * File: AISModelClassifier.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public abstract class AISModelClassifier implements Serializable {

  protected final int kNumNeighbours;

  protected final Normalize normaliser;

  protected final CellPool model;

  protected final AffinityFunction affinityFunction;


  public AISModelClassifier(
    int aKNumNeighbours,
    Normalize aNormalise,
    CellPool aCellPool,
    AffinityFunction aAffinityFunction) {
    normaliser = aNormalise;
    model = aCellPool;
    affinityFunction = aAffinityFunction;

    // adjust knn
    int totalElements = model.getCells().size();
    if (aKNumNeighbours > totalElements) {
      aKNumNeighbours = totalElements;
    }
    kNumNeighbours = aKNumNeighbours;
  }


  public String getModelSummary(Instances aInstances) {
    StringBuffer buffer = new StringBuffer(1024);

    // data reduction percentage
    buffer.append(" - Classifier Statistics - \n");
    double dataReduction = 100.0 * (1.0 - ((double) model.size() / (double) aInstances.numInstances()));
    buffer.append("Data Reduction Percentage:..." + Utils.format.format(dataReduction) + "%\n");
    buffer.append("\n");

    // determine the breakdown of cells
    int numClasses = aInstances.numClasses();
    int[] counts = new int[numClasses];

    for (Cell c : model.getCells()) {
      counts[(int) c.getClassification()]++;
    }
    buffer.append(" - Classifier Memory Cells - \n");
    buffer.append("Total: " + model.getCells().size() + "\n");
    for (int i = 0; i < counts.length; i++) {
      int val = counts[i];
      buffer.append(aInstances.classAttribute().value(i) + ": " + val + "\n");
    }

    return buffer.toString();
  }

  protected int[] determineClassCountForkNN(
    Instance aInstance,
    LinkedList<Cell> affinitySortedCells) {
    int numClasses = aInstance.classAttribute().numValues();
    int[] classCount = new int[numClasses];

    for (int i = 0; i < kNumNeighbours; i++) {
      int classIndex = (int) affinitySortedCells.get(i).getClassification();
      classCount[classIndex]++;
    }

    return classCount;
  }

  public int classifyInstance(Instance aInstance) {
    // normalise vector
    try {
      normaliser.input(aInstance);
    }
    catch (Exception e) {
      throw new RuntimeException("Unable to classify instance: " + e.getMessage(), e);
    }
    aInstance = normaliser.output();

    // classify
    return classify(aInstance);
  }

  protected abstract int classify(Instance aInstance);


  public AffinityFunction getAffinityFunction() {
    return affinityFunction;
  }

  public int getKNumNeighbours() {
    return kNumNeighbours;
  }

  public CellPool getModel() {
    return model;
  }
}
