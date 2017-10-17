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
