package weka.classifiers.neural.lvq.algorithm;

import weka.classifiers.neural.common.RandomWrapper;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.classifiers.neural.lvq.model.CommonModel;
import weka.core.Instance;

/**
 * Description: Implementation of the OLVQ1 algorithm. Uses individual
 * learning rates for each codebook vector to adjust the codebook vectors
 * values
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class Olvq1Algorithm extends LVQAlgorithmAncestor {

  public Olvq1Algorithm(LearningRateKernel aLearningKernel,
			CommonModel aModel,
			RandomWrapper aRand) {
    super(aLearningKernel, aModel, aRand);
    // apply the learning rate to all codebook vectors
    model.applyLearningRateToAllVectors(learningKernel.getInitialLearningRate());
  }

  protected boolean usingGlobalLearningRate() {
    return false; // individual learning rates
  }


  protected void updateModel(Instance aInstance, double aGlobalLearningRate) {
    // reference the bmu
    CodebookVector bmu = model.getBmu(aInstance);
    // adjust the codebook vector
    updateVector(bmu, aInstance, bmu.getIndividualLearningRate());
    // adjust the learning rate
    adjustIndividualLearningRate(bmu, aInstance);
  }
}