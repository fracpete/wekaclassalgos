package weka.classifiers.neural.lvq.initialise;

import weka.classifiers.neural.common.RandomWrapper;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

/**
 * Date: 26/05/2004
 * File: SimpleKMeans.java
 *
 * @author Jason Brownlee
 */
public class SimpleKMeansInitialiser extends ClusterAlgorithmInitialiser {

  public SimpleKMeansInitialiser(RandomWrapper aRand, Instances aInstances, int aNumClusters) {
    super(aRand, aInstances, aNumClusters);
  }

  protected Clusterer getClusterAlgorithm() throws Exception {
    SimpleKMeans algorithm = new SimpleKMeans();
    algorithm.setNumClusters(numClusters);
    algorithm.setSeed((int) rand.getSeed());
    return algorithm;
  }

  protected Instances getClusterCentroids() {
    return ((SimpleKMeans) clusterAlgorithm).getClusterCentroids();
  }
}
