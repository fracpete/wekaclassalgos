package weka.classifiers.immune.immunos;

import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;


/**
 * Type: ImmunosCommon<br>
 * Date: 28/01/2005<br>
 * <br>
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class Immunos1Algorithm implements Serializable {

  protected DistanceFunction affinityFunction;

  protected Instances[] groups;


  protected void prepareAlgorithm(Instances aInstances) {
    // prepare affinity function
    affinityFunction = new DistanceFunction(aInstances);

    // prepare groups
    int numClasses = aInstances.classAttribute().numValues();
    groups = new Instances[numClasses];
    for (int i = 0; i < groups.length; i++) {
      groups[i] = new Instances(aInstances, 0);
    }
  }

  protected void prepareClassifier(Instances aInstances) {
    // process dataset
    for (int i = 0; i < aInstances.numInstances(); i++) {
      Instance current = aInstances.instance(i);
      int classification = (int) current.classValue();
      groups[classification].add(current);
    }
  }

  protected double sumAffinity(Instances aGroup, Instance aInstance) {
    double[] dataInstance = aInstance.toDoubleArray();
    double sumAffinity = 0.0;

    for (int j = 0; j < aGroup.numInstances(); j++) {
      Instance current = aGroup.instance(j);
      double affinity = affinityFunction.distanceEuclideanUnnormalised(current.toDoubleArray(), dataInstance);
      sumAffinity += affinity;
    }

    return sumAffinity;
  }


  protected double bestAffinity(Instances aGroup, Instance aInstance) {
    double[] dataInstance = aInstance.toDoubleArray();
    double bestAffinity = Double.POSITIVE_INFINITY;

    for (int j = 0; j < aGroup.numInstances(); j++) {
      Instance current = aGroup.instance(j);
      double affinity = affinityFunction.distanceEuclideanUnnormalised(current.toDoubleArray(), dataInstance);
      if (affinity < bestAffinity) {
	bestAffinity = affinity;
      }
    }

    return bestAffinity;
  }


  public void train(Instances aInstances)
    throws Exception {
    // prepare algorithm
    prepareAlgorithm(aInstances);

    // prepare classifier
    prepareClassifier(aInstances);
  }


  protected double[] calculateGroupAvidity(Instance aInstance) {
    double[] avidity = new double[groups.length];

    for (int i = 0; i < groups.length; i++) {
      // check for empty group
      if (groups[i].numInstances() == 0) {
	avidity[i] = Double.NaN;
      }
      else {
	// calculate sum affinity
	double sumAffinity = sumAffinity(groups[i], aInstance);
	sumAffinity = (groups[i].numInstances() / sumAffinity);
	avidity[i] = sumAffinity;

	// gets similar reuslts
	//	            double bestAffinity = sumAffinity(groups[i], aInstance);
	//	            bestAffinity = (groups[i].numInstances() / bestAffinity);
	//	            avidity[i] = bestAffinity;
      }
    }

    return avidity;
  }

  public double classify(Instance aInstance) {
    // calculate avidity for each group
    double[] avidity = calculateGroupAvidity(aInstance);

    int bestIndex = -1;
    double bestAvidity = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < avidity.length; i++) {
      if (Double.isNaN(avidity[i])) {
	continue;
      }

      if (avidity[i] > bestAvidity) {
	bestAvidity = avidity[i];
	bestIndex = i;
      }
    }

    return bestIndex;
  }
}
