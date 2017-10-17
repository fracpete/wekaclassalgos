package weka.classifiers.neural.common.transfer;

import java.io.Serializable;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 * @author Jason Brownlee
 * @version 1.0
 */

public abstract class TransferFunction implements Serializable
{
    public final static double UPPER_THREAHOLD = +45.0;
    public final static double LOWER_THREAHOLD = -45.0;


    // from the NN FAQ on overflow protection
    public double overflowProtectionTransfer(double activation)
    {
        double output = 0.0;

        if (activation < LOWER_THREAHOLD)
        {
            output = getMinimum();
        }
        else if (activation > UPPER_THREAHOLD)
        {
            output = getMaximum();
        }
        else
        {
            output = transfer(activation);
        }

       return output;
    }

    public abstract double transfer(double activation);

    public abstract double derivative(double activation, double transferted);

    public abstract double getMaximum();

    public abstract double getMinimum();
}