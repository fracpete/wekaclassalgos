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

import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.classifiers.neural.lvq.vectordistance.AttributeDistance;

/**
 * Date: 25/05/2004
 * File: ModelInitialiser.java
 *
 * @author Jason Brownlee
 */
public interface ModelInitialiser {

  void initialiseCodebookVector(CodebookVector aCodebookVector);

  AttributeDistance[] getAttributeDistanceList();

  String[] getClassLables();
}
