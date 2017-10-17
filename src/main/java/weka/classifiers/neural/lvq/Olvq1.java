package weka.classifiers.neural.lvq;

import weka.classifiers.Evaluation;
import weka.classifiers.neural.common.learning.LearningKernelFactory;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.lvq.algorithm.Olvq1Algorithm;
import weka.core.Instances;

import java.util.Collection;

/**
 * Description: An implementation of the OLVQ1 algorithm for use in WEKA
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class Olvq1 extends LvqAlgorithmAncestor {


  protected void trainModel(Instances instances) {
    // construct the algorithm
    LearningRateKernel learningKernel = LearningKernelFactory.factory(learningFunction, learningRate, trainingIterations);
    Olvq1Algorithm algorithm = new Olvq1Algorithm(learningKernel, model, random);
    // add event listeners
    addEventListenersToAlgorithm(algorithm);
    // train the algorithm
    algorithm.trainModel(instances, trainingIterations);
  }


  /**
   * Validate algorithm specific arguments
   *
   * @throws Exception
   */
  protected void validateArguments() throws Exception {
    // do nothing
  }

  /**
   * Return a list of algorithm specific options
   *
   * @return Collection
   */
  protected Collection getListOptions() {
    // do nothing
    return null;
  }

  protected void setArguments(String[] options)
    throws Exception {
  }

  /**
   * Return a list of algorithm specific options and values
   */
  protected Collection getAlgorithmOptions() {
    // do nothing
    return null;
  }

  /**
   * Return information about this algorithm implementation
   */
  public String globalInfo() {
    StringBuffer buffer = new StringBuffer(100);
    buffer.append("Learning Vector Quantisation (LVQ) - OLVQ1.");
    buffer.append("The same as LVQ1, except each codebook vector has its own learning rate. ");
    buffer.append("If the BMU has the same class, the individual learning rate is increased, ");
    buffer.append("otherwise it is decreased.");
    return buffer.toString();
  }

  /**
   * Entry point into the algorithm for direct usage
   *
   * @param args
   */
  public static void main(String[] args) {
    try {
      System.out.println(Evaluation.evaluateModel(new Olvq1(), args));
    }
    catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }
}