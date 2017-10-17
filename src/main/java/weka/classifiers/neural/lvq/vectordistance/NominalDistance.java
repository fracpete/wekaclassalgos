package weka.classifiers.neural.lvq.vectordistance;



/**
 * 
 * Description: Calculates the distance between two nominal data values
 * 
 * <br>
 * Copyright (c) Jason Brownlee 2004
 * </p>
 * @author Jason Brownlee
 *
 */
public class NominalDistance implements AttributeDistance
{
	/**
	 * Distance between nominal attribute values, lower the distnace the closer the values.
	 * 
	 * @param instanceValue
	 * @param codebookValue
	 * @return
	 */		
    public double distance(double instanceValue, double codebookValue)
    {
		// calculate the difference
		double delta = (instanceValue - codebookValue);
		// square the difference
		return (delta * delta);
		
		//
		// JB 24May2004
		// Note: I don't like this idea of binary comparison - the return value
		// assumes too much about the data, who's to know if 1.0 is meaningful or too meaningful
		//		
        // binary comparison
        //return (instanceValue == codebookValue) ? 0.0 : 1.0;
    }
}