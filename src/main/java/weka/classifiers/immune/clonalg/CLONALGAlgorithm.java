
package weka.classifiers.immune.clonalg;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Random;

import weka.classifiers.immune.affinity.AttributeDistance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Type: CLONALGAlgorithm<br>
 * Date: 19/01/2005<br>
 * <br>
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 */
public class CLONALGAlgorithm implements Serializable
{
    protected final double clonalFactor; // beta
    protected final int antibodyPoolSize; // N
    protected final int selectionPoolSize; // n
    protected final int replacementPoolSize; // d
    protected final int numGenerations; // Ngen
    protected final long seed; // random number seed
    protected final double remainderPoolRatio; // typically 5%-8%
    
    protected Antibody [] memoryPool;
    protected Antibody [] remainderPool;
    protected Random rand;
    protected DistanceFunction affinityFunction;

    /**
     * 
     */
    public CLONALGAlgorithm(
            double aClonalFactor,
            int aAntibodyPoolSize,
            int aSelectionPoolSize,
            int aReplacementPoolSize,
            int aNumGenerations,
            long aSeed,
            double aRemainderPoolRatio)
    {
        clonalFactor = aClonalFactor;
        antibodyPoolSize = aAntibodyPoolSize;
        selectionPoolSize = aSelectionPoolSize;
        replacementPoolSize = aReplacementPoolSize;
        numGenerations = aNumGenerations;
        seed = aSeed;
        remainderPoolRatio = aRemainderPoolRatio;
    }

    
    
    protected void algorithmPreperation(Instances aAntigens)
    {
        rand = new Random(seed);
        int remainderSize = (int) Math.round(antibodyPoolSize * remainderPoolRatio);
        int memorySize = (antibodyPoolSize - remainderSize);
        
        if(remainderSize == 0)
        {
            throw new RuntimeException("Remainder pool size cannot be zero!");
        }
        
        if(memorySize == 0)
        {
            throw new RuntimeException("Memory pool size cannot be zero!");
        }
        
        if(remainderSize < replacementPoolSize)
        {
            throw new RuntimeException("The size of the remainder pool ["+remainderSize+"] is less than the number of elements replaced each iteration ["+replacementPoolSize+"].");
        }
        
        memoryPool = new Antibody[memorySize];
        remainderPool  = new Antibody[remainderSize];
        affinityFunction = new DistanceFunction(aAntigens);
    }
    
    protected void initialiseAntibodyPool(Instances aAntigens)
    {
        aAntigens.randomize(rand);
        
        // populate the remainder pool
        for (int i = 0; i < remainderPool.length; i++)
        {
            remainderPool[i] = new Antibody(aAntigens.instance(rand.nextInt(aAntigens.numInstances())));
        }
        
        // populate the memory pool
        for (int i = 0; i < memoryPool.length; i++)
        {
            memoryPool[i] = new Antibody(aAntigens.instance(rand.nextInt(aAntigens.numInstances())));
        }
    }
    
    protected void train(Instances aAntigens)
    	throws Exception
    {
        // prepare the algorithm
        algorithmPreperation(aAntigens);
       
        // initialise the memory pools
        initialiseAntibodyPool(aAntigens);
        
        for (int i = 0; i < numGenerations; i++)
        {
            // randomise the dataset
            aAntigens.randomize(rand);
            
            for (int j = 0; j < aAntigens.numInstances(); j++)
            {            
	            // select a random antigen without reselection
	            Instance currentInstance = aAntigens.instance(j);
                
	            // calculate affinities for the antibody pool
	            calculateAffinity(remainderPool, currentInstance);
	            calculateAffinity(memoryPool, currentInstance);
	            
	            // locate the best n antibodies
	            Antibody [] bestSet = selectBestAntibodySet(currentInstance);
	            
	            // perform cloning and mutation (maturation)
	            Antibody [] cloneSet = prepareCloneSet(bestSet, currentInstance);
	            
	            // calculate the affinities for the clonal antibody pool
	            calculateAffinity(cloneSet, currentInstance);
	            
	            // select a candidate antigen
	            Arrays.sort(cloneSet);
	            Antibody candidate = cloneSet[0];
	             
	            // check if a replacement can occur
	            // must be of the correct class
	            if(candidate.getClassification() == currentInstance.classValue())
	            {
	                // must have better affinity than best in memory pool
	                Arrays.sort(memoryPool);
	                if(candidate.getAffinity() < memoryPool[0].getAffinity())
	                {
	                    memoryPool[0] = candidate;
	                }
	            }
	            
	            // replace the lower d members of the remainder pool with random instances
	            Arrays.sort(remainderPool);
	            for (int k = cloneSet.length-1; k < replacementPoolSize; k++)
                {
	                remainderPool[k] = generateRandomAntibodyInRange(remainderPool[k], currentInstance);
                }
            }
        }
        
        // the memory pool is used as the classifier
    }
    
    
    public double classify(Instance aInstance)
    {
        // calculate affinity for instance
        calculateAffinity(memoryPool, aInstance);
        // sort by affinity
        Arrays.sort(memoryPool);
        // return the classification of the best match
        return memoryPool[0].getClassification();
    }
    
    
    protected Antibody generateRandomAntibodyInRange(Antibody aAntibody, Instance aInstance)
    {
        // simply mutate the hell out of it
        mutateClone(aAntibody, 1.0, aInstance);        
        // mutate the class 
        double [] data = aAntibody.getAttributes();
        data[aAntibody.getClassIndex()] = rand.nextInt(aInstance.classAttribute().numValues());
        // return the same antibody
        return aAntibody;
    }
    
    protected void mutateClone(
            Antibody aClone, 
            double aMutationRate,
            Instance aInstance)
    {
        double [][] minmax = affinityFunction.getMinMax();
        AttributeDistance[] attribs = affinityFunction.getDistanceMeasures();       
        
        double [] data = aClone.getAttributes();
       
        for (int i = 0; i < data.length; i++)
        {
            if(attribs[i].isClass())
            {
                continue;
            }
            else if(attribs[i].isNominal())
            {
                data[i] = rand.nextInt(aInstance.attribute(i).numValues());
            }
            else if(attribs[i].isNumeric())
            {
                // determine the mutation rate based range
                double range = (minmax[i][1] - minmax[i][0]);
                range = (range * aMutationRate);
                
                // determine bounds for new value based on range
                double min = Math.max(data[i]-(range/2.0), minmax[i][0]);
                double max = Math.min(data[i]+(range/2.0), minmax[i][1]);
                
                // generate new value in VALID range and store
                data[i] = min + (rand.nextDouble() * (max-min));
            }
            else
            {
                throw new RuntimeException("Unsuppored attribute type!");
            }
        }
    }
    
    protected Antibody [] prepareCloneSet(Antibody [] aBestSet, Instance aInstance)
    {
        LinkedList<Antibody> clones = new LinkedList<Antibody>();
        
        // totally rank based
        for (int i = 1; i <= aBestSet.length; i++)       
        {
            int numClones = (int) Math.round((clonalFactor * antibodyPoolSize) / i);
            
            Antibody current = aBestSet[i-1];
            double mutationRate = (double)i / (double)aBestSet.length;
            
            for (int j = 0; j < numClones; j++)
            {
                Antibody a = new Antibody(current); // create
                mutateClone(a, mutationRate, aInstance); // mutate
                clones.add(a); // add to clone list
            }
        }
        
        return clones.toArray(new Antibody[clones.size()]);
    }
    
    
    protected Antibody [] selectBestAntibodySet(Instance aInstance)
    {
        Antibody [] bestSet = new Antibody[selectionPoolSize];
        LinkedList<Antibody> totalSet = new LinkedList<Antibody>();
        
        for (int i = 0; i < remainderPool.length; i++)
        {
            totalSet.add(remainderPool[i]);
        }        
        for (int i = 0; i < memoryPool.length; i++)
        {
            totalSet.add(memoryPool[i]);
        }
        
        // sort ascending in terms of affinity values, though
        // decending in terms of actual affinity quality
        Collections.sort(totalSet);
        
        for (int i = 0; i < bestSet.length; i++)
        {
            bestSet[i] = totalSet.get(i);
        }
        
        return bestSet;
    }
    
    protected void calculateAffinity(Antibody [] antibodies, Instance aInstance)
    {
        double [] data = aInstance.toDoubleArray();
        
        for (int i = 0; i < antibodies.length; i++)
        {
            double affinity = affinityFunction.calculateDistance(antibodies[i].getAttributes(), data);
            antibodies[i].setAffinity(affinity);
        }
    }
    
    
}
