
package weka.classifiers.immune.clonalg;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 * Type: CLONALGTest<br>
 * Date: 19/01/2005<br>
 * <br>
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 */
public class CLONALGTest
{
    public static void main(String[] args)
    {
		try
		{
			// prepare dataset
            Instances dataset = new Instances(new FileReader("data/iris.arff"));
            dataset.setClassIndex(dataset.numAttributes() - 1);
            CLONALG algorithm = new CLONALG();   
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
