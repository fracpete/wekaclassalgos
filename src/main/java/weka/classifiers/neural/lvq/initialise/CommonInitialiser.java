package weka.classifiers.neural.lvq.initialise;

import weka.classifiers.neural.common.RandomWrapper;
import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.classifiers.neural.lvq.vectordistance.AttributeDistance;
import weka.classifiers.neural.lvq.vectordistance.DistanceFactory;
import weka.core.Instances;

/**
 * Date: 25/05/2004
 * File: CommonRandomInitialiser.java
 *
 * @author Jason Brownlee
 */
public abstract class CommonInitialiser implements ModelInitialiser {

  protected final RandomWrapper rand;

  protected final Instances trainingInstances;

  protected final int numClasses;

  protected final int numAttributes;

  protected final int classIndex;

  protected final int totalInstances;

  public CommonInitialiser(RandomWrapper aRand, Instances aInstances) {
    rand = aRand;
    trainingInstances = aInstances;
    numClasses = trainingInstances.classAttribute().numValues();
    totalInstances = trainingInstances.numInstances();
    classIndex = trainingInstances.classIndex();
    numAttributes = trainingInstances.numAttributes();
  }

  public void initialiseCodebookVector(CodebookVector aCodebookVector) {
    double[] attributes = getAttributes();

    // repace any missing values
    for (int j = 0; j < attributes.length; j++) {
      if (weka.core.Utils.isMissingValue(attributes[j])) {
	// replace with a random double - shown to produce better results
	// because it assumes nothing about the data
	attributes[j] = rand.getRand().nextDouble();
      }
    }

    // initialise the codebook vector
    aCodebookVector.initialise(attributes, classIndex, numClasses);
  }

  public abstract double[] getAttributes();

  public AttributeDistance[] getAttributeDistanceList() {
    return DistanceFactory.getAttributeDistanceList(trainingInstances);
  }

  public String[] getClassLables() {
    String[] classLabels = new String[numClasses];

    // cache each class double value at its index
    for (int i = 0; i < classLabels.length; i++) {
      classLabels[i] = trainingInstances.classAttribute().value(i);
    }

    return classLabels;
  }

  protected int makeRandomSelection(int aTotalChoices) {
    // get random number
    int selection = rand.getRand().nextInt();
    // max positive
    selection = Math.abs(selection);
    // transform to within required bounds
    selection = (selection % aTotalChoices);
    return selection;
  }
}
