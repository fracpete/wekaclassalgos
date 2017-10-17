package weka.classifiers.neural.common.initialisation;

import weka.classifiers.neural.common.RandomWrapper;


/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 * @author Jason Brownlee
 * @version 1.0
 */

public class Initialisation
{
    public static void initialiseVectorToRandom(double [] vector,
                                         		double upper,
                                         		double lower,
												RandomWrapper rand)
    {
        for(int i=0; i<vector.length; i++)
        {
            vector[i] = getRandomDouble(upper, lower, rand);
        }
    }

    public static void initialiseVectorToRandomWithSign(double [] vector,
                                                 		double upper,
                                                 		double lower,
                                                 		RandomWrapper rand)
    {
        for(int i=0; i<vector.length; i++)
        {
            vector[i] = getRandomDoubleWithSign(upper, lower, rand);
        }
    }



    // Generate random double between the two specified ranges.
    // Ranges limited to 0.0 amd 1.0
    public static double getRandomDouble(double upperRange,
                                         double lowerRange,
										 RandomWrapper rand)
    {
        return lowerRange + (rand.getRand().nextDouble() * (upperRange - lowerRange));
    }



    // Generate a random double between the two specified ranges.
    // Ranges are limited to 0.0 and 1.0, the number produced will be randomly
    // either positive or negative (between -1.0 and +1.0)
    public static double getRandomDoubleWithSign(double upperRange,
                                                 double lowerRange,
                                                 RandomWrapper rand)
    {
        double value = getRandomDouble(lowerRange, upperRange, rand);

        // 50% chance of a negative
        if(rand.getRand().nextBoolean())
        {
            return -value;
        }

        return value;
    }

}