/*
 * Created on 15/01/2005
 *
 */
package weka.classifiers.immune.airs;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 * Type: SampleAIRSUsage<br>
 * File: SampleAIRSUsage.java<br>
 * Date: 15/01/2005<br>
 * <br>
 * Description: 
 * <br>
 * @author Jason Brownlee
 *
 */
public class SampleAIRSUsage
{

	public static void main(String[] args)
	{
		try
		{
			// prepare dataset
            Instances dataset = new Instances(new FileReader("data/iris.arff"));
            dataset.setClassIndex(dataset.numAttributes() - 1);
            AIRS2 algorithm = new AIRS2();   
        	// evaulate
            Evaluation evaluation = new Evaluation(dataset);
            evaluation.crossValidateModel(algorithm, dataset, 10, new Random(1));            
            // print algorithm details
            System.out.println(algorithm.toString());
            // print stats
            System.out.println(evaluation.toSummaryString());
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
}
