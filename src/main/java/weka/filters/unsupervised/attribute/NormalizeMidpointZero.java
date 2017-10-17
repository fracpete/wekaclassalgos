package weka.filters.unsupervised.attribute;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.UnsupervisedFilter;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 * @author Jason Brownlee
 * @version 1.0
 */

public class NormalizeMidpointZero extends Filter
    implements UnsupervisedFilter
{
    public final static double MAX = +1.0;
    public final static double MIN = -1.0;


    /** The minimum values for numeric attributes. */
    private double[] m_MinArray;

    /** The maximum values for numeric attributes. */
    private double[] m_MaxArray;




    protected double normaliseToMidpointZero(double value,
                                             double max,
                                             double min)
    {
        // calculate the mid range for the data
        double midrange = (max + min) / 2.0;

        // calculate the range for the data set
        double range = (max - min);

        // normalise to midpoint zero with range of 2 (-1.0 to +1.0)
        return (value - midrange) / (range / 2.0);

    }



    /**
     * Sets the format of the input instances.
     *
     * @param instanceInfo an Instances object containing the input
     * instance structure (any instances contained in the object are
     * ignored - only the structure is required).
     * @return true if the outputFormat may be collected immediately
     * @exception Exception if the input format can't be set
     * successfully
     */
    public boolean setInputFormat(Instances instanceInfo)
        throws Exception
    {

        super.setInputFormat(instanceInfo);
        setOutputFormat(instanceInfo);
        m_MinArray = m_MaxArray = null;
        return true;
    }

    /**
     * Input an instance for filtering. Filter requires all
     * training instances be read before producing output.
     *
     * @param instance the input instance
     * @return true if the filtered instance may now be
     * collected with output().
     * @exception IllegalStateException if no input format has been set.
     */
    public boolean input(Instance instance)
    {

        if (getInputFormat() == null)
        {
            throw new IllegalStateException("No input instance format defined");
        }
        if (m_NewBatch)
        {
            resetQueue();
            m_NewBatch = false;
        }
        if (m_MinArray == null)
        {
            bufferInput(instance);
            return false;
        }
        else
        {
            convertInstance(instance);
            return true;
        }
    }

    /**
     * Signify that this batch of input to the filter is finished.
     * If the filter requires all instances prior to filtering,
     * output() may now be called to retrieve the filtered instances.
     *
     * @return true if there are instances pending output
     * @exception IllegalStateException if no input structure has been defined
     */
    public boolean batchFinished()
    {
        if (getInputFormat() == null)
        {
            throw new IllegalStateException("No input instance format defined");
        }

        if (m_MinArray == null)
        {
            Instances input = getInputFormat();

            // Compute minimums and maximums
            m_MinArray = new double[input.numAttributes()];
            m_MaxArray = new double[input.numAttributes()];

            for (int i = 0; i < input.numAttributes(); i++)
            {
                m_MinArray[i] = Double.NaN;
            }

            for (int j = 0; j < input.numInstances(); j++)
            {
                double[] value = input.instance(j).toDoubleArray();
                for (int i = 0; i < input.numAttributes(); i++)
                {
                    if (input.attribute(i).isNumeric())
                    {
                        if (!weka.core.Utils.isMissingValue(value[i]))
                        {
                            if (Double.isNaN(m_MinArray[i]))
                            {
                                m_MinArray[i] = m_MaxArray[i] = value[i];
                            }
                            else
                            {
                                if (value[i] < m_MinArray[i])
                                {
                                    m_MinArray[i] = value[i];
                                }
                                if (value[i] > m_MaxArray[i])
                                {
                                    m_MaxArray[i] = value[i];
                                }
                            }
                        }
                    }
                }
            }

            // Convert pending input instances
            for (int i = 0; i < input.numInstances(); i++)
            {
                convertInstance(input.instance(i));
            }
        }
        // Free memory
        flushInput();

        m_NewBatch = true;
        return (numPendingOutput() != 0);
    }

    /**
     * Convert a single instance over. The converted instance is
     * added to the end of the output queue.
     *
     * @param instance the instance to convert
     */
    private void convertInstance(Instance instance)
    {
        Instance inst = null;

        if (instance instanceof SparseInstance)
        {
            double[] newVals = new double[instance.numAttributes()];
            int[] newIndices = new int[instance.numAttributes()];
            double[] vals = instance.toDoubleArray();
            int ind = 0;

            for (int j = 0; j < instance.numAttributes(); j++)
            {
                double value;

                // value is numeric and has a value
                if (instance.attribute(j).isNumeric() && (!weka.core.Utils.isMissingValue(vals[j])) )
                {
                    // not a number or all values are the same
                    if (Double.isNaN(m_MinArray[j]) || (m_MaxArray[j] == m_MinArray[j]))
                    {
                        value = MIN; // set to normalized min
                    }
                    else
                    {
                        // normalize the value
                        value = normaliseToMidpointZero(vals[j], m_MaxArray[j], m_MinArray[j]);
                    }

                    if (value != MIN) // not the smallest value
                    {
                        newVals[ind] = value;
                        newIndices[ind] = j;
                        ind++;
                    }
                }
                else
                {
                    value = vals[j];

                    if (value != MIN)
                    {
                        newVals[ind] = value;
                        newIndices[ind] = j;
                        ind++;
                    }
                }
            }

            double[] tempVals = new double[ind];
            int[] tempInd = new int[ind];
            System.arraycopy(newVals, 0, tempVals, 0, ind);
            System.arraycopy(newIndices, 0, tempInd, 0, ind);
            inst = new SparseInstance(instance.weight(), tempVals, tempInd,
                                      instance.numAttributes());
        }
        else
        {
            double[] vals = instance.toDoubleArray();
            for (int j = 0; j < getInputFormat().numAttributes(); j++)
            {
                // is numeric and not a missing value
                if (instance.attribute(j).isNumeric() && (!weka.core.Utils.isMissingValue(vals[j])))
                {
                    // not a number of the values are the same
                    if (Double.isNaN(m_MinArray[j]) || (m_MaxArray[j] == m_MinArray[j]))
                    {
                        vals[j] = MIN; // set to normalized min
                    }
                    else
                    {
                        // normalize the value
                        vals[j] = normaliseToMidpointZero(vals[j], m_MaxArray[j], m_MinArray[j]);
                    }
                }
            }

            inst = new DenseInstance(instance.weight(), vals);
        }
        inst.setDataset(instance.dataset());
        push(inst);
    }

    /**
     * Main method for testing this class.
     *
     * @param argv should contain arguments to the filter:
     * use -h for help
     */
    public static void main(String[] argv)
    {

        try
        {
            if (Utils.getFlag('b', argv))
            {
                Filter.batchFilterFile(new NormalizeMidpointZero(), argv);
            }
            else
            {
                Filter.filterFile(new NormalizeMidpointZero(), argv);
            }
        }
        catch (Exception ex)
        {
            System.out.println(ex.getMessage());
        }
    }

}