package weka.classifiers.neural.common.training;

import java.util.Enumeration;

import weka.classifiers.neural.common.NeuralModel;
import weka.classifiers.neural.common.RandomWrapper;
import weka.core.Instance;
import weka.core.Instances;


/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 * @author Jason Brownlee
 * @version 1.0
 */

public class OnlineTrainer extends NeuralTrainer
{
	public OnlineTrainer(RandomWrapper aRand)
	{
		super(aRand);
	}	

    public void trainModel(NeuralModel aModel,
                           Instances aInstances,
                           int numIterations)
    {
        Instances epochInstances = new Instances(aInstances);

        // train until we can stop
        for(int iteration=0; iteration<numIterations; iteration++)
        {
            // prepare the model for an epoch
			aModel.startingEpoch();
            
			// get the learning rate
			double learingRate = aModel.getLearningRate(iteration);

            // randomize the dataset
            epochInstances.randomize(rand.getRand());

            // perform a single epoch
            Enumeration e = epochInstances.enumerateInstances();
            while(e.hasMoreElements())
            {
                // get an instance
                Instance instance = (Instance) e.nextElement();

                // update the model for a given instance
				aModel.updateModel(instance, learingRate);
            }

            // finished epoch
			aModel.finishedEpoch(epochInstances, learingRate);
        }
    }
}