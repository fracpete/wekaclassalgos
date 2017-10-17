package weka.classifiers.neural.lvq.initialise;

import weka.classifiers.neural.lvq.model.CodebookVector;
import weka.classifiers.neural.lvq.vectordistance.AttributeDistance;

/**
 * Date: 25/05/2004
 * File: ModelInitialiser.java
 *
 * @author Jason Brownlee
 */
public interface ModelInitialiser {

  void initialiseCodebookVector(CodebookVector aCodebookVector);

  AttributeDistance[] getAttributeDistanceList();

  String[] getClassLables();
}
