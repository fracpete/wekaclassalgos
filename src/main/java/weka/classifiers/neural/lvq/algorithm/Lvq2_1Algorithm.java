package weka.classifiers.neural.lvq.algorithm;

import weka.classifiers.neural.common.RandomWrapper;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.classifiers.neural.lvq.model.CommonModel;
import weka.classifiers.neural.lvq.model.LvqModel;
import weka.core.Instance;

/**
 * 
 * Description: Implementation of the LVQ2.1 algorithm. Makes use of a
 * window size to determine when the top 2 BMU's can be adjusted.
 * 
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 * @author Jason Brownlee
 *
 */
public class Lvq2_1Algorithm extends Lvq1Algorithm
{	
	
    protected final double windowSize;
    
    
	
	public Lvq2_1Algorithm(LearningRateKernel aLearningKernel,						
						   CommonModel aModel,
						   RandomWrapper aRand,
						   double aWindow)
	{
		super(aLearningKernel, aModel, aRand);
		windowSize = aWindow;
	}
	
	

	/**
	 * Responsbile for updating the model for the given data instance.
	 * The top 2 BMU's are returned, and only updated if they are both in the same class
	 * and the distance is within 1.0 the window size
	 * 
	 * @param aInstance
	 * @param lrate
	 */
    protected void updateModel(Instance aInstance,
                               double lrate)
    {
        // get bmus
        CodebookVector [] bmus = ((LvqModel)model).get2Bmu(aInstance);

		// both bmu's must have different classes, one must have the correct 
		// class and the distance ratio must be within the window
		if(bmusOfDifferentClassesAndInWindow(bmus[0], bmus[1], aInstance))
        {
            // adjust the codebook vector
            updateVector(bmus[0], aInstance, lrate);
			updateVector(bmus[1], aInstance, lrate);
        }
    }
	/**
	 * Checks that the two provided codebook vectors are of a different class,
	 * and that one of them has the same class as the data instance. Also checks
	 * that the ration of the vectors distance is within a defined window
	 * 
	 * @param bmu1
	 * @param bmu2
	 * @param aInstance
	 * @return
	 */
	protected boolean bmusOfDifferentClassesAndInWindow(CodebookVector bmu1, 
														CodebookVector bmu2, 
														Instance aInstance)
	{
		// both bmu's must have different classes
		if(!isSameClass(bmu1, bmu2))
		{
			// one of the bmu's classes must match the class of the instance
			if(isSameClass(aInstance, bmu1) || isSameClass(aInstance, bmu2))
			{
				// min (di/dj, dj/di) > s, where s = (1-w)/(1+w)
				double distanceRatio = bmu1.getDistance() / bmu2.getDistance();
				double window = (1.0 - windowSize) / (1.0 + windowSize);				
				
				// vectors must be within the window
				if(distanceRatio > window)
				{
					return true;
				}
			}
		}
		
		return false;
	}      
}