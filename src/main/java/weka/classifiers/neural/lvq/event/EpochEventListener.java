package weka.classifiers.neural.lvq.event;

import weka.classifiers.neural.lvq.model.CommonModel;

/**
 * Date: 28/05/2004
 * File: EpochEventListener.java
 *
 * @author Jason Brownlee
 */
public interface EpochEventListener {

  void epochEvent(int aEpochNumber, int totalEpochs, CommonModel aModel);
}
