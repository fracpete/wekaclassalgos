package weka.classifiers.neural.lvq.algorithm;

import weka.classifiers.neural.common.RandomWrapper;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.classifiers.neural.lvq.model.CommonModel;
import weka.classifiers.neural.lvq.model.LvqModel;
import weka.core.Instance;

/**
 * Description: Represents an implementation of the LVQ algorithm
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class Lvq3Algorithm extends Lvq2_1Algorithm {

  protected final double epsilonRate;


  public Lvq3Algorithm(LearningRateKernel aLearningKernel,
		       CommonModel aModel,
		       RandomWrapper aRand,
		       double aWindow,
		       double aEpsilonRate) {
    super(aLearningKernel, aModel, aRand, aWindow);
    epsilonRate = aEpsilonRate;
  }


  protected void updateModel(Instance aInstance, double lrate) {
    // calculate distances to all codebook vectors
    CodebookVector[] bmus = ((LvqModel) model).get2Bmu(aInstance);

    // both bmu's must have different classes, one must have the correct
    // class and the distance ratio must be within the window
    if (bmusOfDifferentClassesAndInWindow(bmus[0], bmus[1], aInstance)) {
      // adjust the codebook vector
      updateVector(bmus[0], aInstance, lrate);
      updateVector(bmus[1], aInstance, lrate);
    }
    // both bmu's are of the same class and match the expected class
    else if (bmusOfCorrectClass(bmus[0], bmus[1], aInstance)) {
      // adjusted learing rate
      double adjustedLrate = (lrate * epsilonRate);
      updateVector(bmus[0], aInstance, adjustedLrate);
      updateVector(bmus[1], aInstance, adjustedLrate);
    }
  }

  /**
   * Checks that the two provided codebook vectors are of the same class, and
   * both have the same class as the data instance
   *
   * @param bmu1
   * @param bmu2
   * @param aInstance
   * @return
   */
  protected boolean bmusOfCorrectClass(CodebookVector bmu1,
				       CodebookVector bmu2,
				       Instance aInstance) {
    if (isSameClass(bmu1, bmu2) && isSameClass(aInstance, bmu1)) {
      return true;
    }

    return false;
  }
}