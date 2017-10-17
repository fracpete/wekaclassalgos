package weka.classifiers.neural.lvq.topology;

import weka.core.Tag;

/**
 * Date: 25/05/2004
 * File: NeighbourhoodDistanceFactory.java
 * 
 * @author Jason Brownlee
 *
 */
public class NeighbourhoodDistanceFactory
{
	public final static int NEIGHBOURHOOD_DISTANCE_RECTANGLE = 1;
	public final static int NEIGHBOURHOOD_DISTNACE_HEXAGONAL = 2;
	
	public final static Tag [] TAGS_MODEL_TOPOLOGY =
	{
		 new Tag(NEIGHBOURHOOD_DISTANCE_RECTANGLE, "Rectangular"),
		 new Tag(NEIGHBOURHOOD_DISTNACE_HEXAGONAL, "Hexagonal")       
	};
	
	public final static String DESCRIPTION;
	
	static
	{
		StringBuffer buffer = new StringBuffer();
		buffer.append("(");		
		
		for (int i = 0; i < TAGS_MODEL_TOPOLOGY.length; i++)
		{
			buffer.append(TAGS_MODEL_TOPOLOGY[i].getID());
			buffer.append("==");
			buffer.append(TAGS_MODEL_TOPOLOGY[i].getReadable());			
			
			if(i != TAGS_MODEL_TOPOLOGY.length-1)
			{
				buffer.append(", ");
			}
		}
		buffer.append(")");
		
		DESCRIPTION = buffer.toString();
	}
	
	public final static NeighbourhoodDistance factory(int aNeighbourhoodDistance)
	{
		NeighbourhoodDistance distance = null;
		
		switch(aNeighbourhoodDistance)
		{
			case NEIGHBOURHOOD_DISTANCE_RECTANGLE:
			{
				distance = new RectangleNeighborhoodDistance();
				break;
			}
			case NEIGHBOURHOOD_DISTNACE_HEXAGONAL:
			{
				distance = new HexagonalNeighbourhoodDistance();
				break;
			}			
			default:
			{
				throw new RuntimeException("Unknown neighbourhood distance: " + aNeighbourhoodDistance);
			}
		}
		
		return distance;
	}
}
