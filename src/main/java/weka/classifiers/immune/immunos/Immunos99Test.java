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
