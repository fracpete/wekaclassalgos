
package weka.classifiers.immune.airs.algorithm;

import weka.core.Instances;

/**
 * Type: AISTrainer<br>
 * Date: 7/01/2005<br>
 * <br>
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 */
public interface AISTrainer
{
	AISModelClassifier train(Instances aInstances) throws Exception;
	
	String getTrainingSummary();
}
