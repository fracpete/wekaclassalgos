package weka.classifiers.neural.lvq.model;

/**
 * Date: 25/05/2004
 * File: ModelUpdater.java
 * 
 * @author Jason Brownlee
 *
 */
public interface ModelUpdater
{
	void updateCodebookVector(CodebookVector aCodebookVector);
}
