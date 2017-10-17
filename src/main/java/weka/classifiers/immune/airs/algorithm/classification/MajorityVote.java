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
package weka.classifiers.immune.airs.algorithm.classification;

import weka.classifiers.immune.airs.algorithm.AISModelClassifier;
import weka.classifiers.immune.airs.algorithm.AffinityFunction;
import weka.classifiers.immune.airs.algorithm.Cell;
import weka.classifiers.immune.airs.algorithm.CellPool;
import weka.core.Instance;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.LinkedList;

/**
 * Type: MajorityVote
 * File: MajorityVote.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class MajorityVote extends AISModelClassifier {

  public MajorityVote(
    int aKNumNeighbours,
    Normalize aNormalise,
    CellPool aCellPool,
    AffinityFunction aAffinityFunction) {
    super(aKNumNeighbours, aNormalise, aCellPool, aAffinityFunction);
  }


  protected int classify(Instance aInstance) {
    // respond to affinity
    LinkedList<Cell> cells = model.affinityResponseUnnormalised(aInstance, affinityFunction);
    // determine the majority for the top k cells
    int[] classCounts = determineClassCountForkNN(aInstance, cells);

    int largestIndex = 0;
    int largestCount = classCounts[0];

    for (int i = 1; i < classCounts.length; i++) {
      if (classCounts[i] > largestCount) {
	largestCount = classCounts[i];
	largestIndex = i;
      }
    }

    return largestIndex;
  }
}
