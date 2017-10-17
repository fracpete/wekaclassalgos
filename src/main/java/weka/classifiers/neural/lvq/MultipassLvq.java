/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package weka.classifiers.neural.lvq;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.neural.lvq.model.CommonModel;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Vector;

/**
 * Date: 26/05/2004
 * File: MultipassLvq2.java
 *
 * @author Jason Brownlee
 */
public class MultipassLvq extends AbstractClassifier
  implements WeightedInstancesHandler {

  private final static int PARAM_CLASSIFIER_1 = 0;

  private final static int PARAM_CLASSIFIER_2 = 1;

  private final static String[] PARAMETERS =
    {
      "A", // pass 1
      "B",  // pass 2
    };

  private final static String[] PARAMETER_NOTES =
    {
      "<algorithm for pass 1>", // pass 1
      "<algorithm for pass 2>",  // pass 2
    };

  private final static String[] PARAM_DESCRIPTIONS =
    {
      "LVQ algorithm and parameters used for the first pass. The OLVQ1 algorithm is recommended by Kohonen. "
	+ "Typically this pass is 50 times the number of codebook vectors.",
      "LVQ algorithm and parameters used for the second pass. "
	+ "The second pass should be a long fine tuning pass with LVQ1, LVQ2.1 or LVQ3. Typically this pass is 10 times the number of iterations. "
    };


  private final static int TOTAL_PASSES = 2;

  protected LvqAlgorithmAncestor[] algorithms;

  protected long[] trainingTimes;

  protected boolean modelIsInitialised;


  public MultipassLvq() {
    algorithms = new LvqAlgorithmAncestor[TOTAL_PASSES];
    trainingTimes = new long[TOTAL_PASSES];

    algorithms[0] = new Olvq1();
    algorithms[0].setTotalTrainingIterations(algorithms[0].getTotalCodebookVectors() * 50);
    algorithms[0].setLearningRate(0.3);

    algorithms[1] = new Lvq3();
    algorithms[1].setTotalTrainingIterations(algorithms[0].getTotalCodebookVectors() * 50 * 10);
    algorithms[1].setLearningRate(0.05);
  }


  public void buildClassifier(Instances instances)
    throws Exception {
    // validate user provided arguments
    validateAlgorithmArguments();

    // construct and train each model
    for (int i = 0; i < algorithms.length; i++) {
      trainingTimes[i] = System.currentTimeMillis();
      // check for first pass
      if (i == 0) {
	// build the model
	algorithms[i].setPrepareBmuStatistis(false); // not used
      }
      // not the first pass
      else {
	// pre-initialise the algorithm
	algorithms[i].setPreInitialisedModel(algorithms[0].getModel());
	// check for last pass
	if (i == algorithms.length - 1) {
	  // want to use statistics from the last pass
	  algorithms[i].setPrepareBmuStatistis(true);
	}
      }
      algorithms[i].setDebug(m_Debug);
      algorithms[i].buildClassifier(instances);

      trainingTimes[i] = (System.currentTimeMillis() - trainingTimes[i]);
    }
  }

  public double[] distributionForInstance(Instance instance)
    throws Exception {
    // use the first algorithm to classify the instance
    // all algorithms use the same underlying model to classify at this point
    return algorithms[0].distributionForInstance(instance);
  }

  public String globalInfo() {
    StringBuffer buffer = new StringBuffer(100);
    buffer.append("Learning Vector Quantisation - Multipass LVQ, where the same underlying model is tuned by two LVQ algorithms. ");
    buffer.append("The is the recommended usage for LVQ as described by Kohonoen. ");
    buffer.append("The same model is used, only it is constructed by passing through two ");
    buffer.append("LVQ algorithms. It is recommended that the first pass is rough (eg. OLVQ1), ");
    buffer.append("and that the second pass is used to fine tune the model (eg. LVQ1, LVQ2.1 or LVQ3).\n\n");
    buffer.append("It is important to understand that the first pass will construct the model ");
    buffer.append("that is fine tuned by the second pass. This means that algorithm parameters ");
    buffer.append("in the second pass used to initialise the model will not be used. ");
    buffer.append("These include: initialisation mode, total codebook vectors and use voting.");

    return buffer.toString();
  }

  public String toString() {
    // all stats are pulled about the model from the last pass (their all valid)

    StringBuffer buffer = new StringBuffer();
    AlgorithmAncestor lastPass = (AlgorithmAncestor) algorithms[algorithms.length - 1];

    if (super.m_Debug) {
      // bmu hits report
      if (lastPass.prepareBmuStatistis) {
	buffer.append(lastPass.prepareTrainingBMUReport());
	buffer.append("\n");
      }

      // class distributions for each codebook vector
      buffer.append(lastPass.prepareIndividualClassDistributionReport());
      buffer.append("\n");

      // quantisation error
      buffer.append(lastPass.quantisationErrorReport());
      buffer.append("\n");

      // codebook vectors
      buffer.append(lastPass.prepareCodebookVectorReport());
      buffer.append("\n");
    }

    // build times
    buffer.append(prepareTrainingTimeReport());
    buffer.append("\n");

    // distribution report
    buffer.append(lastPass.prepareClassDistributionReport("-- Cass Distribution --"));
    buffer.append("\n");

    return buffer.toString();
  }


  public String prepareTrainingTimeReport() {
    StringBuffer buffer = new StringBuffer(1024);
    buffer.append("-- Training Time Breakdown --\n");
    long total = 0;
    for (int i = 0; i < trainingTimes.length; i++) {
      buffer.append("Pass " + i + ": " + trainingTimes[i] + "ms\n");
      total += trainingTimes[i];
    }
    buffer.append("Total Model Preparation Time: " + total + "ms\n");
    return buffer.toString();
  }

  protected void validateAlgorithmArguments()
    throws Exception {
    for (int i = 0; i < algorithms.length; i++) {
      if (algorithms[i] == null) {
	throw new Exception("Algorithm for pass " + (i + 1) + " was not defined.");
      }
    }
  }

  public Enumeration listOptions() {
    Vector list = new Vector(PARAMETERS.length);

    for (int i = 0; i < PARAMETERS.length; i++) {
      String param = "-" + PARAMETERS[i] + " " + PARAMETER_NOTES[i];
      list.add(new Option("\t" + PARAM_DESCRIPTIONS[i], PARAMETERS[i], 1, param));
    }

    return list.elements();
  }

  public void setOptions(String[] options)
    throws Exception {
    for (int i = 0; i < PARAMETERS.length; i++) {
      String data = Utils.getOption(PARAMETERS[i].charAt(0), options);

      if (data == null || data.length() == 0) {
	continue;
      }

      switch (i) {
	case PARAM_CLASSIFIER_1: {
	  setPass1LVQClassifier(prepareClassifierFromParameterString(data));
	  break;
	}
	case PARAM_CLASSIFIER_2: {
	  setPass2LVQClassifier(prepareClassifierFromParameterString(data));
	  break;
	}
	default: {
	  throw new Exception("Invalid option offset: " + i);
	}
      }
    }
  }


  private Classifier prepareClassifierFromParameterString(String s)
    throws Exception {
    String[] classifierSpec = null;
    String classifierName = null;

    // split the string into its componenets
    classifierSpec = Utils.splitOptions(s);

    // verify some componets were specified
    if (classifierSpec.length == 0) {
      throw new Exception("Invalid classifier specification string");
    }

    // copy the name, then clear it from the list (it will not be a valid param for itself)
    classifierName = classifierSpec[0];
    classifierSpec[0] = "";

    // consrtuct the classifier with its params
    return AbstractClassifier.forName(classifierName, classifierSpec);
  }

  public String[] getOptions() {
    LinkedList list = new LinkedList();

    list.add("-" + PARAMETERS[PARAM_CLASSIFIER_1]);
    list.add(getClassifierSpec(algorithms[0]));

    list.add("-" + PARAMETERS[PARAM_CLASSIFIER_2]);
    list.add(getClassifierSpec(algorithms[1]));

    return (String[]) list.toArray(new String[list.size()]);
  }

  protected String getClassifierSpec(Classifier c) {
    String name = c.getClass().getName();
    String params = Utils.joinOptions(((LvqAlgorithmAncestor) c).getOptions());
    return name + " " + params;
  }

  public String pass1LVQClassifierTipText() {
    return PARAM_DESCRIPTIONS[PARAM_CLASSIFIER_1];
  }

  public String pass2LVQClassifierTipText() {
    return PARAM_DESCRIPTIONS[PARAM_CLASSIFIER_2];
  }

  public void setPass1LVQClassifier(Classifier c) {
    if (!(c instanceof LvqAlgorithmAncestor)) {
      throw new IllegalArgumentException("Only single pass LVQ algorithms can be specified.");
    }

    algorithms[0] = (LvqAlgorithmAncestor) c;
  }

  public Classifier getPass1LVQClassifier() {
    return algorithms[0];
  }

  public void setPass2LVQClassifier(Classifier c) {
    if (!(c instanceof LvqAlgorithmAncestor)) {
      throw new IllegalArgumentException("Only single pass LVQ algorithms can be specified.");
    }

    algorithms[1] = (LvqAlgorithmAncestor) c;
  }

  public Classifier getPass2LVQClassifier() {
    return algorithms[1];
  }

  public CommonModel getModel() {
    return algorithms[0].getModel();
  }


  public static void main(String[] args) {
    try {
      System.out.println(Evaluation.evaluateModel(new MultipassLvq(), args));
    }
    catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }
}
