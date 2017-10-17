/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm;

import weka.core.Instance;

import java.io.Serializable;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;

/**
 * Type: AISModel
 * File: AISModel.java
 * Date: 30/12/2004
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class CellPool implements Serializable {

  public final static AffinityComparator affinityComparator = new AffinityComparator();

  public final static StimulationComparator stimulationComparator = new StimulationComparator();

  public final static ResourceComparator resourceComparator = new ResourceComparator();


  protected final LinkedList<Cell> cells;


  public CellPool(LinkedList<Cell> aCells) {
    cells = aCells;
  }


  public LinkedList<Cell> affinityResponseNormalised(Instance aInstance, AffinityFunction affinity) {
    // calculate affinity for all cells
    double[] features = aInstance.toDoubleArray();
    for (Cell c : cells) {
      double aff = affinity.affinityNormalised(features, c);
      c.setAffinity(aff);
    }
    // sort by affinity
    Collections.sort(cells, affinityComparator);
    // return sorted cells (most similar to least similar)
    return cells;
  }

  public LinkedList<Cell> affinityResponseUnnormalised(Instance aInstance, AffinityFunction affinity) {
    // calculate affinity for all cells
    double[] features = aInstance.toDoubleArray();
    for (Cell c : cells) {
      double aff = affinity.affinityUnnormalised(features, c);
      c.setAffinity(aff);
    }
    // sort by affinity
    Collections.sort(cells, affinityComparator);
    // return sorted cells (most similar to least similar)
    return cells;
  }

  public LinkedList<Cell> resourceResponse() {
    // sort by resources
    Collections.sort(cells, resourceComparator);
    return cells;
  }

  public static class AffinityComparator implements Comparator<Cell> {

    /**
     * Compare cells based on affinity. The lower the value the
     * higher the affinity.
     * <p>
     * Orders cells in ascending order, meaning the highest affinity
     * members are at the beginning of the array
     *
     * @param o1
     * @param o2
     * @return
     */
    public int compare(Cell o1, Cell o2) {
      if (o1.affinity < o2.affinity) {
	return -1;
      }
      else if (o1.affinity > o2.affinity) {
	return +1;
      }

      return 0;
    }
  }

  public static class StimulationComparator implements Comparator<Cell> {

    /**
     * Compare cells based on stimulation. The higher the value the
     * higher the stimulation.
     * <p>
     * Orders cells in decending order, meaning the highest stimulation
     * members are at the beginning of the array
     *
     * @param o1
     * @param o2
     * @return
     */
    public int compare(Cell o1, Cell o2) {
      if (o1.stimulation > o2.stimulation) {
	return -1;
      }
      else if (o1.stimulation < o2.stimulation) {
	return +1;
      }

      return 0;
    }
  }

  public static class ResourceComparator implements Comparator<Cell> {

    /**
     * Compare cells based on numResources.
     * <p>
     * Orders cells in decending order, meaning the most resources
     * members are at the beginning of the array
     *
     * @param o1
     * @param o2
     * @return
     */
    public int compare(Cell o1, Cell o2) {
      if (o1.numResources > o2.numResources) {
	return -1;
      }
      else if (o1.numResources < o2.numResources) {
	return +1;
      }

      return 0;
    }
  }


  public void add(LinkedList<Cell> aNewList) {
    cells.addAll(aNewList);
  }

  public void add(Cell aCell) {
    cells.add(aCell);
  }

  public void delete(Cell aCell) {
    cells.remove(aCell);
  }

  public boolean isEmpty() {
    return cells.isEmpty();
  }

  public int size() {
    return cells.size();
  }


  public LinkedList<Cell> getCells() {
    return cells;
  }

  public Iterator<Cell> iterator() {
    return cells.iterator();
  }

  public void orderByResources() {
    Collections.sort(cells, resourceComparator);
  }

  public void orderByStimulation() {
    Collections.sort(cells, stimulationComparator);
  }

  public void orderByAffinity() {
    Collections.sort(cells, affinityComparator);
  }
}
