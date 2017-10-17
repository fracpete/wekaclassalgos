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
 * Created on 8/01/2005
 *
 */
package weka.classifiers.immune.airs.algorithm;

import weka.core.Instances;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.LinkedList;


/**
 * Type: MemoryCellMerger<br>
 * File: MemoryCellMerger.java<br>
 * Date: 8/01/2005<br>
 * <br>
 * Description:
 * <br>
 *
 * @author Jason Brownlee
 */
public interface MemoryCellMerger {

  AISModelClassifier mergeMemoryCells(
    LinkedList<Cell>[] cells,
    int aKNN,
    Normalize aNormalise,
    AffinityFunction aFunction,
    Instances aDataset);
}
