package weka.classifiers.neural.lvq.model;

import weka.core.Instance;


/**
 * Description: Represents a LVQ model generated using a varient of the LVQ algorithm
 * for a dataset
 * <p>
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 *
 * @author Jason Brownlee
 */
public class LvqModel extends CommonModel {

  public LvqModel(int totalVectors) {
    super(totalVectors);
  }


  public CodebookVector[] get2Bmu(Instance aInstance) {
    double[] instance = aInstance.toDoubleArray();
    double tmp = 0.0;

    double[] distances = new double[2];
    CodebookVector[] bmus = new CodebookVector[2];

    // set the best as the first codebook
    bmus[0] = codebookCollection[0];
    distances[0] = distance(instance, codebookCollection[0].getAttributes(), Double.POSITIVE_INFINITY);

    // calculate second best
    if ((tmp = distance(instance, codebookCollection[1].getAttributes(), distances[0])) < distances[0]) {
      // first best becomes second best
      bmus[1] = bmus[0];
      distances[1] = distances[0];

      // second element becomes first best
      bmus[0] = codebookCollection[1];
      distances[0] = tmp;
    }
    else {
      // second element is second best
      bmus[1] = codebookCollection[1];
      distances[1] = tmp;
    }

    // process all codebook vectors
    for (int i = 2; i < codebookCollection.length; i++) {
      double distance = distance(instance, codebookCollection[i].getAttributes(), distances[1]);

      // check if better than second best
      if (distance < distances[1]) {
	// check if better than the best
	if (distance < distances[0]) // new best and new second best
	{
	  // best becomes second best
	  distances[1] = distances[0];
	  bmus[1] = bmus[0];

	  // current becomes the best
	  distances[0] = distance;
	  bmus[0] = codebookCollection[i];
	}
	else // new second best
	{
	  distances[1] = distance;
	  bmus[1] = codebookCollection[i];
	}
      }
    }

    // store the distances in the bmus
    bmus[0].setBmuHit(distances[0], aInstance);
    bmus[1].setBmuHit(distances[1], aInstance);

    return bmus;
  }


}