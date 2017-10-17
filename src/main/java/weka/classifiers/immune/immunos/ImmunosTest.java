package weka.classifiers.immune.immunos;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

/**
 * Type: CLONALGTest<br>
 * Date: 19/01/2005<br>
 * <br>
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class ImmunosTest {

  public final static NumberFormat format = new DecimalFormat();

  public static void main(String[] args) {
    String[] problems =
      {
	//            "data/heart-c.arff",
	//            "data/iris.arff",
	//            "data/sonar.arff",
	"data/breast-w.arff"
      };
    int[] folds =
      {
	//                10,
	//                3,
	//                13,
	10
      };
    int totalTests = 10;

    try {
      double[] results = new double[totalTests];
      double[] datareduction = new double[totalTests];

      for (int p = 0; p < problems.length; p++) {
	int totalFolds = folds[p];
	String filename = problems[p];

	// load the dataset
	Instances dataset = new Instances(new FileReader(filename));
	dataset.setClassIndex(dataset.numAttributes() - 1);

	for (int i = 0; i < totalTests; i++) {
	  long seed = i + 1;

	  // evaulate
	  Classifier algorithm = getClassifier();
	  Evaluation evaluation = new Evaluation(dataset);
	  evaluation.crossValidateModel(algorithm, dataset, totalFolds, new Random(seed));

	  // record results
	  results[i] = evaluation.pctCorrect();
	  datareduction[i] = ((Immunos99) algorithm).getDataReduction();
	  System.out.println(format.format(results[i]) + "%, " + format.format(datareduction[i]) + "%");
	}

	String all = arrayToString(results);
	System.out.println("All: " + all);
	System.out.println("Result Accuracy (mean,stdv,min,max): " + getStatistic(results));
	System.out.println("Result Reduction: " + getStatistic(datareduction));
      }
    }
    catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static String arrayToString(double[] v) {
    StringBuffer b = new StringBuffer();

    for (int i = 0; i < v.length; i++) {
      b.append(format.format(v[i]) + "%");
      if (i != v.length - 1) {
	b.append(", ");
      }
    }

    return b.toString();
  }

  protected static Classifier getClassifier() {
    //    	Classifier c = new Immunos1();
    //    	Classifier c = new Immunos2();

    Immunos99 c = new Immunos99();
    c.setMinimumFitnessThreshold(-1);
    c.setTotalGenerations(1);
    c.setSeedPopulationPercentage(0.2);

    return c;
  }


  protected static String getStatistic(double[] data) {
    double mean = mean(data);
    double stdev = stdev(data, mean);
    double min = min(data);
    double max = max(data);

    //return format.format(mean) + "% " + "(" + format.format(stdev) + "), min "+format.format(min)+"%, max "+format.format(max)+"%";
    return format.format(mean) + "%, " + format.format(stdev) + ", " + format.format(min) + "%, " + format.format(max) + "%";
  }

  protected static double min(double[] results) {
    double min = Double.POSITIVE_INFINITY;

    for (int i = 0; i < results.length; i++) {
      if (results[i] < min) {
	min = results[i];
      }
    }

    return min;
  }

  protected static double max(double[] results) {
    double max = Double.NEGATIVE_INFINITY;

    for (int i = 0; i < results.length; i++) {
      if (results[i] > max) {
	max = results[i];
      }
    }

    return max;
  }

  protected static double mean(double[] results) {
    double mean = 0.0;
    double sum = 0.0;
    for (int i = 0; i < results.length; i++) {
      sum += results[i];
    }
    mean = (sum / results.length);
    return mean;
  }

  protected static double stdev(double[] results, double mean) {
    // standard deviation -
    // square root of the average squared deviation from the mean
    double stdev = 0.0;
    double sum = 0.0;
    for (int i = 0; i < results.length; i++) {
      double diff = mean - results[i];
      sum += diff * diff;
    }
    stdev = Math.sqrt(sum / results.length);
    return stdev;
  }
}
