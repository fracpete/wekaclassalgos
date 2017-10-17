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
package weka.classifiers.immune.airs.algorithm.initialisation;

import weka.classifiers.immune.airs.algorithm.Cell;
import weka.classifiers.immune.airs.algorithm.ModelInitialisation;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

/**
 * Type: RandomInstancesInitialisation
 * File: RandomInstancesInitialisation.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class RandomInstancesInitialisation extends ModelInitialisation {

  public RandomInstancesInitialisation(Random aRand) {
    super(aRand);
  }

  /**
   * @param aInstances
   * @return
   */
  public Cell generateCell(Instances aInstances) {
    int selection = rand.nextInt(aInstances.numInstances());
    Instance inst = aInstances.instance(selection);
    return new Cell(inst);
  }

}
