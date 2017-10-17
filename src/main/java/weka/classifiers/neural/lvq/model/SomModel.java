package weka.classifiers.neural.lvq.model;

import weka.classifiers.neural.lvq.topology.NeighbourhoodDistance;

/**
 * Date: 25/05/2004
 * File: SOMModel.java
 *
 * @author Jason Brownlee
 */
public class SomModel extends CommonModel {

  protected final NeighbourhoodDistance neighbourhoodDistance;

  protected final int mapWidth;

  protected final int mapHeight;


  public SomModel(NeighbourhoodDistance aNeighbourhoodDistance,
		  int aMapWidth,
		  int aMapHeight) {
    super(aMapWidth * aMapHeight);
    neighbourhoodDistance = aNeighbourhoodDistance;
    mapWidth = aMapWidth;
    mapHeight = aMapHeight;
  }


  public double calculateNeighbourhoodDistance(CodebookVector aBmu, CodebookVector aVector) {
    // determine vector rectangular coordinates
    int bx = aBmu.getId() % mapWidth;
    int by = aBmu.getId() / mapWidth;
    int tx = aVector.getId() % mapWidth;
    int ty = aVector.getId() / mapWidth;
    // calculate distance
    return neighbourhoodDistance.neighborhoodDistance(bx, by, tx, ty);
  }
}
