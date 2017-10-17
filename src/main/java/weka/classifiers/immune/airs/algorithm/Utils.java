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

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

/**
 * Type: Utils
 * File: Utils.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public final class Utils {

  public final static NumberFormat format = new DecimalFormat();


  public final static boolean isSameClass(Instance aInstance, Cell aCell) {
    return aInstance.classValue() == aCell.getClassification();
  }


  public final static double calculateAffinityThreshold(
    Instances aInstances,
    int affinityThresholdNumInstances,
    Random rand,
    AffinityFunction affinityFunction) {
    Instances newset = new Instances(aInstances);

    // check if all should be used
    if (affinityThresholdNumInstances < 1 || affinityThresholdNumInstances > newset.numInstances()) {
      affinityThresholdNumInstances = newset.numInstances();
    }
    // prune some
    else if (newset.numInstances() > affinityThresholdNumInstances) {
      // randomise the dataset
      newset.randomize(rand);

      while (newset.numInstances() > affinityThresholdNumInstances) {
	newset.delete(0);
      }
    }

    int totalInstances = newset.numInstances();
    double sumAffinity = 0.0;
    int count = 0;

    // sum affinity values
    for (int i = 0; i < totalInstances; i++) {
      Instance first = newset.instance(i);

      for (int j = i + 1; j < totalInstances; j++) {
	sumAffinity += affinityFunction.affinityNormalised(first, newset.instance(j));
	count++;
      }
    }

    // take the mean
    return sumAffinity / count;
  }


  public final static int performPrunning(
    CellPool aMemoryPool,
    Instances instances,
    AffinityFunction affinityFunction) {
    LinkedList<Cell> cells = aMemoryPool.getCells();
    int totalPruned = 0;

    // clear usage
    for (Cell c : cells) {
      c.clearUsage();
    }

    // calculate usage
    for (int i = 0; i < instances.numInstances(); i++) {
      Cell best = aMemoryPool.affinityResponseNormalised(instances.instance(i), affinityFunction).getFirst();
      best.incrementUsage();
    }

    // remove all without usage
    for (Iterator<Cell> iter = cells.iterator(); iter.hasNext(); ) {
      Cell element = iter.next();
      if (element.getUsage() == 0) {
	iter.remove();
	totalPruned++;
      }
    }

    return totalPruned;
  }

}
