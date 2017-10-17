
package weka.classifiers.immune.airs;

import weka.classifiers.Classifier;

/**
 * Type: AIRS1Test
 * Date: 6/01/2005
 * 
 * 
 * @author Jason Brownlee
 */
public class AIRS2ParallelTest extends AIRSAlgorithmTester
{
	public static void main(String[] args)
	{
		try
		{
			AIRSAlgorithmTester tester = new AIRS2ParallelTest();
			tester.run();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	
	protected void setSeed(Classifier aClassifier, long aSeed)
	{
		((AIRS2Parallel)aClassifier).setSeed(aSeed);
	}
	
	protected Classifier getAIRSAlgorithm()
	{
		AIRS2Parallel algorithm = new AIRS2Parallel();
		return algorithm;
	}
}
