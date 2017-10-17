/*
 * Created on 15/01/2005
 *
 */
package weka.classifiers.immune.airs;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.immune.airs.algorithm.Utils;
import weka.core.Instances;

/**
 * Type: AIRSAlgorithmTester<br>
 * File: AIRSAlgorithmTester.java<br>
 * Date: 15/01/2005<br>
 * <br>
 * Description: 
 * <br>
 * @author Jason Brownlee
 *
 */
public abstract class AIRSAlgorithmTester
{
	private final static int CROSSVALIDATION = 10;
	private final static int NUM_TESTS = 10;

	public static final String [] filenames = {
		"data/iris.arff", 
		"data/diabetes.arff", 
		"data/sonar.arff", 
		"data/balance-scale.arff",
		"data/breast-w.arff",
		"data/heart-c.arff",
		"data/ionosphere.arff",
		};	

	public void run()
		throws Exception
	{		
		for (int i = 0; i < filenames.length; i++)
		{
            Instances dataset = new Instances(new FileReader(filenames[i]));
            dataset.setClassIndex(dataset.numAttributes() - 1);

            double[] results = new double[NUM_TESTS];
            for (int j = 0; j < NUM_TESTS; j++)
            {
                // prepare the algorithm
            	Classifier algorithm = getAIRSAlgorithm();
            	long seed = j+1;
            	setSeed(algorithm, seed);
            	// evaulate
                Evaluation evaluation = new Evaluation(dataset);
                long start = System.currentTimeMillis();
                evaluation.crossValidateModel(algorithm, dataset, CROSSVALIDATION, new Random(seed));
                long end = System.currentTimeMillis();
                results[j] = evaluation.pctCorrect();
                
                double time = ((end-start)/1000.0);
                System.out.println((j+1)+" - Correct("+Utils.format.format(results[j])+"%), Time("+Utils.format.format(time)+"sec).");                
            }

            // mean
            double mean = mean(results);            
            // standard deviation
            double stdev = stdev(results, mean);
            // print details
            StringBuffer buffer = new StringBuffer();
            buffer.append(filenames[i]+": ");
            buffer.append("Accuracy - "+Utils.format.format(mean)+"% ("+Utils.format.format(stdev)+")");
            System.out.println(buffer.toString());
		}
	}
	
	protected double mean(double [] results)
	{
        double mean = 0.0;
        double sum = 0.0;
        for (int i = 0; i < results.length; i++)
        {
            sum += results[i];
        }
        mean = (sum / results.length);
        return mean;
	}
	
	protected double stdev(double [] results, double mean)
	{
        // standard deviation - 
		// square root of the average squared deviation from the mean
        double stdev = 0.0;
        double sum = 0.0;
        for (int i = 0; i < results.length; i++)
        {
        	double diff = mean - results[i];
        	sum += diff * diff;
        }
        stdev = Math.sqrt(sum / results.length);
        return stdev;
	}
	
	
	protected abstract void setSeed(Classifier aClassifier, long aSeed);
	
	protected abstract Classifier getAIRSAlgorithm();
}
