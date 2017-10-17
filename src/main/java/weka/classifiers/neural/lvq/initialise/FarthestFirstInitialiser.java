package weka.classifiers.neural.lvq.initialise;

import weka.classifiers.neural.common.RandomWrapper;
import weka.clusterers.Clusterer;
import weka.clusterers.FarthestFirst;
import weka.core.Instances;

/**
 * Date: 26/05/2004
 * File: FarthestFirstInitialiser.java
 *
 * @author Jason Brownlee
 */
public class FarthestFirstInitialiser extends ClusterAlgorithmInitialiser {

  public FarthestFirstInitialiser(RandomWrapper aRand, Instances aInstances, int aNumClusters) {
    super(aRand, aInstances, aNumClusters);
  }

  protected Clusterer getClusterAlgorithm() throws Exception {
    UserFriendlyFarthestFirst algorithm = new UserFriendlyFarthestFirst();
    algorithm.setNumClusters(numClusters);
    algorithm.setSeed((int) rand.getSeed());
    return algorithm;
  }

  protected Instances getClusterCentroids() {
    return ((UserFriendlyFarthestFirst) clusterAlgorithm).getClusterCentroids();
  }

  private class UserFriendlyFarthestFirst extends FarthestFirst {

    public Instances getClusterCentroids() {
      return super.m_ClusterCentroids;
    }
  }
}
