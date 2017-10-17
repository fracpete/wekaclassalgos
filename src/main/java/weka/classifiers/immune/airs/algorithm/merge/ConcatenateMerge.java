/*
 * Created on 8/01/2005
 */
package weka.classifiers.immune.airs.algorithm.merge;

import weka.classifiers.immune.airs.algorithm.AISModelClassifier;
import weka.classifiers.immune.airs.algorithm.AffinityFunction;
import weka.classifiers.immune.airs.algorithm.Cell;
import weka.classifiers.immune.airs.algorithm.CellPool;
import weka.classifiers.immune.airs.algorithm.MemoryCellMerger;
import weka.classifiers.immune.airs.algorithm.classification.MajorityVote;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.LinkedList;

/**
 * Type: ConcatonateMerge <br>
 * File: ConcatonateMerge.java <br>
 * Date: 8/01/2005 <br>
 * <br>
 * Description: <br>
 *
 * @author Jason Brownlee
 */
public class ConcatenateMerge implements MemoryCellMerger {

  /**
   * @param cells
   * @return
   */
  public AISModelClassifier mergeMemoryCells(
    LinkedList[] cells,
    int aKNN,
    Normalize aNormalise,
    AffinityFunction aFunction,
    Instances aDataset) {
    LinkedList<Cell> masterList = new LinkedList<Cell>();

    for (int i = 0; i < cells.length; i++) {
      masterList.addAll(cells[i]);
    }

    CellPool pool = new CellPool(masterList);
    MajorityVote classifier = new MajorityVote(aKNN, aNormalise, pool, aFunction);
    return classifier;
  }


}
