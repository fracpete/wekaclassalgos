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

package weka.classifiers.immune.airs;

import weka.classifiers.Classifier;

/**
 * Type: AIRS1Test
 * Date: 6/01/2005
 *
 * @author Jason Brownlee
 */
public class AIRS1Test extends AIRSAlgorithmTester {

  public static void main(String[] args) {
    try {
      AIRSAlgorithmTester tester = new AIRS1Test();
      tester.run();
    }
    catch (Exception e) {
      e.printStackTrace();
    }
  }

  protected void setSeed(Classifier aClassifier, long aSeed) {
    ((AIRS1) aClassifier).setSeed(aSeed);
  }

  protected Classifier getAIRSAlgorithm() {
    AIRS1 algorithm = new AIRS1();
    return algorithm;
  }
}
