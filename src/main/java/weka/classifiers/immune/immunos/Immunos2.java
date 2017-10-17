
package weka.classifiers.immune.immunos;

import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.UnsupportedClassTypeException;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Normalize;

/**
 * Type: Immunos<br>
 * Date: 28/01/2005<br>
 * <br>
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 */
public class Immunos2 extends Classifier implements OptionHandler
{
    protected Immunos2Algorithm algorithm;
    
    protected Normalize normaliser;
    
    
    public void buildClassifier(Instances data) throws Exception
    {
		Instances trainingInstances = new Instances(data);

		
		// must have a class assigned
		if (trainingInstances.classIndex() < 0)
		{
			throw new Exception("No class attribute assigned to instances");
		}		
		// class must be nominal
		else if(!trainingInstances.classAttribute().isNominal())
		{
			throw new UnsupportedClassTypeException("Class attribute must be nominal");
		}
		// must have attributes besides the class attribute
		else if(trainingInstances.numAttributes() <= +1)
		{
			throw new Exception("Dataset contains no supported comparable attributes");
		}
		
		// delete with missing class
		trainingInstances.deleteWithMissingClass();
		for (int i = 0; i < trainingInstances.numAttributes(); i++)
		{
			trainingInstances.deleteWithMissing(i);
		}
		
		// must have some training instances
		if (trainingInstances.numInstances() == 0)
		{
			throw new Exception("No usable training instances!");
		}
		
		// normalise the dataset
        normaliser = new Normalize();
        normaliser.setInputFormat(trainingInstances);
        trainingInstances = Filter.useFilter(trainingInstances, normaliser);

		// construct trainer
		algorithm = new Immunos2Algorithm();

		// train
		algorithm.train(trainingInstances);
    }
    
    public double classifyInstance(Instance instance) throws Exception
    {
        if (algorithm == null)
        {
            throw new Exception("Algorithm has not been prepared.");
        }
        
        // normalise the instance
        normaliser.input(instance);
        normaliser.batchFinished();
        instance = normaliser.output();
        
        return algorithm.classify(instance);
    }

    public String toString()
    {
        StringBuffer buffer = new StringBuffer(1000);
        buffer.append("Immunos-2 v1.0.\n");
        return buffer.toString();
    }

    public String globalInfo()
    {
        StringBuffer buffer = new StringBuffer(1000);
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
    
    
    public Enumeration listOptions()
    {
        Vector<Option> list = new Vector<Option>(15);

        // add parents options
        Enumeration e = super.listOptions();
        while (e.hasMoreElements())
        {
            list.add((Option) e.nextElement());
        }

        return list.elements();
    }

    public void setOptions(String[] options) throws Exception
    {
        // parental option setting
        super.setOptions(options);
    }

    protected double getDouble(String param, String[] options) throws Exception
    {
        String value = Utils.getOption(param.charAt(0), options);
        if (value == null)
        {
            throw new Exception("Parameter not provided: " + param);
        }

        return Double.parseDouble(value);
    }

    protected int getInteger(String param, String[] options) throws Exception
    {
        String value = Utils.getOption(param.charAt(0), options);
        if (value == null)
        {
            throw new Exception("Parameter not provided: " + param);
        }

        return Integer.parseInt(value);
    }

    protected long getLong(String param, String[] options) throws Exception
    {
        String value = Utils.getOption(param.charAt(0), options);
        if (value == null)
        {
            throw new Exception("Parameter not provided: " + param);
        }

        return Long.parseLong(value);
    }


    public String[] getOptions()
    {
        LinkedList<String> list = new LinkedList<String>();

        String[] options = super.getOptions();
        
        for (int i = 0; i < options.length; i++)
        {
            list.add(options[i]);
        }

        return list.toArray(new String[list.size()]);
    }
    
    
    public static void main(String[] argv)
    {

        try
        {
            System.out.println(Evaluation.evaluateModel(new Immunos2(), argv));
        }
        catch (Exception e)
        {
            System.err.println(e.getMessage());
        }
    }
}
