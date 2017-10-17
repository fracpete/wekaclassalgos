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

/*
 * Created on 15/01/2005
 *
 */
package weka.classifiers.immune.airs;

/**
 * Type: AIRSParameterDocumentation<br>
 * File: AIRSParameterDocumentation.java<br>
 * Date: 15/01/2005<br>
 * <br>
 * Description:
 * <br>
 *
 * @author Jason Brownlee
 */
public interface AIRSParameterDocumentation {

  public final static String PARAM_SEED =
    "Random number seed. " +
      "The seed used in for random number generator.";

  public final static String PARAM_ATS =
    "Affinity threshold scalar (ATS). " +
      "Used with the system calculated affinity threshold to determine " +
      "whether or not a candidate memory cell can replace the previous best " +
      "matching memory cell. This occurs if the affinity between the candidate " +
      "and the best match cell is < (AT * ATS).";

  public final static String PARAM_CLONAL_RATE =
    "Clonal rate. " +
      "Used to determine the number of mutated clones to create of an ARB during " +
      "the ARB refinement stage. Calculated as (stimulation * clonal rate).";

  public final static String PARAM_HMR =
    "Hypermutation rate. " +
      "Used with the clonal rate to determine the number of clones a best matching " +
      "memory cell can create to then seed the ARB pool with. This is calculated as " +
      "(stimulation * clonal rate * hypermutation rate).";

  public final static String PARAM_MUTATION_RATE =
    "Mutation rate of cloned ARBs. " +
      "Used to determine the degree of mutation a cloned ARB is subjected to. " +
      "Must be in the range of [0,1].";

  public final static String PARAM_RESOURCES =
    "Total allocatable resources. " +
      "Specifies the maximum number of resources (B-cells) that can be allocated " +
      "to ARBs in the ARB pool. Those ARBs with the weakest stimulation are removed " +
      "from the pool until the total allocated resources is less than the maximum allowable " +
      "resources.";

  public final static String PARAM_STIMULATION =
    "Stimulation threshold. " +
      "Used to determine when to stop refining the pool of ARBs for an antigen. This " +
      "occurs when the mean normalised ARB stimulation value is >= the stimulation threshold. " +
      "Must be in the range of [0,1].";

  public final static String PARAM_AT_INSTANCES =
    "Total training instances to calculate affinity threshold (AT). " +
      "Specifies the number of trainign data instances used to calculate the " +
      "affinity threshold (AT) which is the mean affinity between data instances. " +
      "A value of -1 indicates to use the entire training dataset.";

  public final static String PARAM_ARB_INSTANCES =
    "Initial ARB cell pool size. " +
      "Specifies the number of randomly selected training data instances used to " +
      "seed the ARB cell pool. This paramter must be in the range [0, num training instances].";

  public final static String PARAM_MEM_INSTANCES =
    "Initial memory cell pool size. " +
      "Specifies the number of randomly selected training data instances used to " +
      "seed the memory cell pool. This paramter must be in the range [0, num training instances].";

  public final static String PARAM_KNN =
    "k-Nearest Neighbour. " +
      "Specifies the number of best match memory cells used during the classification " +
      "stage to majority vote hte classification of unseen data instances.";

  public final static String PARAM_THREADS =
    "Number of partitions. " +
      "Specifies the number of partitions that the training dataset is broken into " +
      "and thus the number of threads used to train the classifier.";

  public final static String PARAM_MERGE =
    "Memory cell pool merge mode. " +
      "At the end of training all partitions, the memory cell pools that are prepared " +
      "are merged into a single master memory cell pool. This paramter allows different " +
      "methods for creating the master memory cell pool.";
}
