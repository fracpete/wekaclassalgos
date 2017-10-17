package weka.classifiers.neural.singlelayerperceptron;

import weka.classifiers.Evaluation;
import weka.classifiers.neural.common.NeuralModel;
import weka.classifiers.neural.common.SimpleNeuron;
import weka.classifiers.neural.common.learning.LearningKernelFactory;
import weka.classifiers.neural.common.learning.LearningRateKernel;
import weka.classifiers.neural.common.training.TrainerFactory;
import weka.classifiers.neural.common.transfer.TransferFunction;
import weka.classifiers.neural.common.transfer.TransferFunctionFactory;
import weka.classifiers.neural.singlelayerperceptron.algorithm.WidrowHoffAlgorithm;
import weka.core.Instances;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public class WidrowHoff extends Perceptron {

  public WidrowHoff() {
    // set static values
    transferFunction = TransferFunctionFactory.TRANSFER_STEP; // must be step
    trainingMode = TrainerFactory.TRAINER_ONLINE;

    // set good initial values
    trainingIterations = 500;
    biasInput = SimpleNeuron.DEFAULT_BIAS_VALUE;
    learningRate = 0.1;
    learningRateFunction = LearningKernelFactory.LEARNING_FUNCTION_LINEAR;
    randomNumberSeed = 0;
  }

  protected NeuralModel prepareAlgorithm(Instances instances) throws Exception {
    // prepare the transfer function
    TransferFunction function = TransferFunctionFactory.factory(transferFunction);
    // prepare the learning rate function
    LearningRateKernel lrateFunction = LearningKernelFactory.factory(learningRateFunction, learningRate, trainingIterations);
    // construct the algorithm
    WidrowHoffAlgorithm algorithm = new WidrowHoffAlgorithm(function, biasInput, rand, lrateFunction, instances);
    return algorithm;
  }

  public String globalInfo() {
    StringBuffer buffer = new StringBuffer();

    buffer.append("Single Layer Perceptron : Perceptron Learning Rule, Binary inputs, Step transfer function");

    return buffer.toString();
  }

  /**
   * Entry point into the algorithm for direct usage
   *
   * @param args
   */
  public static void main(String[] args) {
    try {
      System.out.println(Evaluation.evaluateModel(new WidrowHoff(), args));
    }
    catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }
}