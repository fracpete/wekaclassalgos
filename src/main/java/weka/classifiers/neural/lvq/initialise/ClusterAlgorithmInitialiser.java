package weka.classifiers.neural.lvq.initialise;

import weka.classifiers.neural.common.RandomWrapper;
import weka.clusterers.Clusterer;
import weka.core.Instances;

/**
 * Date: 26/05/2004
 * File: CommonClusterAlgorithmInitialiser.java
 *
 * @author Jason Brownlee
 */
public abstract class ClusterAlgorithmInitialiser extends CommonInitialiser {

  protected Clusterer clusterAlgorithm;

  protected Instances clusterCentroids;

  protected int currentCentroid;

  protected int numClusters;

  protected int centroidWrapCounter;

  public ClusterAlgorithmInitialiser(RandomWrapper aRand, Instances aInstances, int aNumClusters) {
    super(aRand, aInstances);
    numClusters = aNumClusters;
    prepareClusterAlgorithm();
  }

  protected abstract Clusterer getClusterAlgorithm() throws Exception;

  protected abstract Instances getClusterCentroids();

  protected void prepareClusterAlgorithm() {
    try {
      // initialise the algorithm
      clusterAlgorithm = getClusterAlgorithm();
      // build the model
      clusterAlgorithm.buildClusterer(trainingInstances);
      // get cluster centroids
      clusterCentroids = getClusterCentroids();
    }
    catch (Exception e) {
      throw new RuntimeException("Failed to preapre cluster algorithm: " + e.getMessage(), e);
    }
  }


  public double[] getAttributes() {
    double[] attributes = clusterCentroids.instance(currentCentroid++).toDoubleArray();

    // check if the index needs to be wrapped, in the even that the cluster
    // algorithm produces less centroids than codebook vectors
    if (currentCentroid >= clusterCentroids.numInstances() - 1) {
      currentCentroid = 0; // reset
      centroidWrapCounter++;
      //System.out.println(" > Wrapped centroid initialisation: " + centroidWrapCounter);
    }

    return attributes;
  }
}
