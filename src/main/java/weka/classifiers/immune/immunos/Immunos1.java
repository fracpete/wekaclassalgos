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

package weka.classifiers.immune.immunos;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Normalize;

/**
 * Type: Immunos<br>
 * Date: 28/01/2005<br>
 * <br>
 * <p>
 * Description:
 *
 * @author Jason Brownlee
 */
public class Immunos1 extends AbstractClassifier {

  protected Immunos1Algorithm algorithm;

  protected Normalize normaliser;

  /**
   * Returns the Capabilities of this classifier.
   *
   * @return the capabilities of this object
   * @see Capabilities
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    result.disableAll();

    // attributes
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.NOMINAL_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    result.setMinimumNumberInstances(1);

    return result;
  }

  public void buildClassifier(Instances data) throws Exception {
    Instances trainingInstances = new Instances(data);
    trainingInstances.deleteWithMissingClass();

    getCapabilities().testWithFail(trainingInstances);

    // normalise the dataset
    normaliser = new Normalize();
    normaliser.setInputFormat(trainingInstances);
    trainingInstances = Filter.useFilter(trainingInstances, normaliser);

    // construct trainer
    algorithm = new Immunos1Algorithm();

    // train
    algorithm.train(trainingInstances);
  }

  public double classifyInstance(Instance instance) throws Exception {
    if (algorithm == null) {
      throw new Exception("Algorithm has not been prepared.");
    }

    // normalise the instance
    normaliser.input(instance);
    normaliser.batchFinished();
    instance = normaliser.output();

    return algorithm.classify(instance);
  }

  public String toString() {
    StringBuilder buffer = new StringBuilder();
    buffer.append("Immunos-1 v1.0.\n");
    return buffer.toString();
  }

  public String globalInfo() {
    StringBuilder buffer = new StringBuilder();
    buffer.append(toString());
    buffer.append("\n\n");

    buffer.append(" Jason Brownlee.  " +
      "[Technical Report].  " +
      "Immunos-81 - The Misunderstood Artificial Immune System.  " +
      "Victoria, Australia: " +
      "Centre for Intelligent Systems and Complex Processes (CISCP), " +
      "Faculty of Information and Communication Technologies (ICT), " +
      "Swinburne University of Technology; " +
      "2005 Feb; " +
      "Technical Report ID: 3-01. ");

    return buffer.toString();
  }


  public static void main(String[] args) {
   runClassifier(new Immunos1(), args);
  }
}
