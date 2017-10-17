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
