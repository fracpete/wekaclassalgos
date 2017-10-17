/*
 * Created on 8/01/2005
 *
 */
package weka.classifiers.immune.airs.algorithm;

import weka.core.Instances;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.LinkedList;


/**
 * Type: MemoryCellMerger<br>
 * File: MemoryCellMerger.java<br>
 * Date: 8/01/2005<br>
 * <br>
 * Description:
 * <br>
 *
 * @author Jason Brownlee
 */
public interface MemoryCellMerger {

  AISModelClassifier mergeMemoryCells(
    LinkedList<Cell>[] cells,
    int aKNN,
    Normalize aNormalise,
    AffinityFunction aFunction,
    Instances aDataset);
}
