
package weka.classifiers.immune.airs;

import weka.classifiers.Classifier;

/**
 * Type: AIRS1Test
 * Date: 6/01/2005
 * 
 * 
 * @author Jason Brownlee
 */
public class AIRS1Test extends AIRSAlgorithmTester
{

	public static void main(String[] args)
	{
		try
		{
			AIRSAlgorithmTester tester = new AIRS1Test();
			tester.run();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	
	protected void setSeed(Classifier aClassifier, long aSeed)
	{
		((AIRS1)aClassifier).setSeed(aSeed);
	}
	
	protected Classifier getAIRSAlgorithm()
	{
		AIRS1 algorithm = new AIRS1();
		return algorithm;
	}
}
