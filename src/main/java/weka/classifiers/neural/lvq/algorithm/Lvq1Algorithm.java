package weka.classifiers.neural.lvq.algorithm;

import weka.classifiers.neural.common.RandomWrapper;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.classifiers.neural.lvq.model.CommonModel;
import weka.core.Instance;

/**
 * Description: Implementation of the LVQ algorithm used to construct a model
 * for a given dataset
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class Lvq1Algorithm extends LVQAlgorithmAncestor {

  public Lvq1Algorithm(LearningRateKernel aLearningKernel,
		       CommonModel aModel,
		       RandomWrapper aRand) {
    super(aLearningKernel, aModel, aRand);
  }


  protected boolean usingGlobalLearningRate() {
    return true;
  }

  protected void updateModel(Instance aInstance,
			     double currentLearningRate) {
    // reference the bmu
    CodebookVector bmu = model.getBmu(aInstance);
    // adjust the codebook vector
    updateVector(bmu, aInstance, currentLearningRate);
  }
}