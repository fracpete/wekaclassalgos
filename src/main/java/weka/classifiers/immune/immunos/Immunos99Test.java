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

package weka.classifiers.immune.immunos;

import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.FileReader;
import java.util.Random;

/**
 * Type: Immunos99Test<br>
 * Date: 19/01/2005<br>
 * <br>
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class Immunos99Test {

  public static void main(String[] args) {
    try {
      String filename = "data/heart-c.arff";
      //			String filename = "data/iris.arff";
      //			String filename = "data/sonar.arff";
      //			String filename = "data/breast-w.arff";

      //			String filename = "data/diabetes.arff";
      //			String filename = "data/balance-scale.arff";

      // prepare dataset
      Instances dataset = new Instances(new FileReader(filename));
      dataset.setClassIndex(dataset.numAttributes() - 1);
      Immunos99 algorithm = new Immunos99();
      // evaulate
      Evaluation evaluation = new Evaluation(dataset);
      evaluation.crossValidateModel(algorithm, dataset, 10, new Random(1));
      // print algorithm details
      System.out.println(algorithm.toString());
      // print stats
      System.out.println(evaluation.toSummaryString());
    }
    catch (Exception e) {
      e.printStackTrace();
    }
  }
}
