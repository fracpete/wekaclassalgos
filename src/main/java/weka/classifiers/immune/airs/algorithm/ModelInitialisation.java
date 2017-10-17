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

import weka.core.Instances;

import java.util.LinkedList;
import java.util.Random;

/**
 * Type: ModelInitialisation
 * File: ModelInitialisation.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public abstract class ModelInitialisation {

  protected final Random rand;

  public ModelInitialisation(Random aRand) {
    rand = aRand;
  }


  public abstract Cell generateCell(Instances aInstances);


  public Cell[] generateCellsArray(Instances aInstances, int numToGenerate) {
    Cell[] all = new Cell[numToGenerate];
    for (int i = 0; i < all.length; i++) {
      all[i] = generateCell(aInstances);
    }
    return all;
  }

  public LinkedList<Cell> generateCellsList(Instances aInstances, int numToGenerate) {
    LinkedList<Cell> all = new LinkedList<Cell>();
    for (int i = 0; i < numToGenerate; i++) {
      all.add(generateCell(aInstances));
    }
    return all;
  }
}
