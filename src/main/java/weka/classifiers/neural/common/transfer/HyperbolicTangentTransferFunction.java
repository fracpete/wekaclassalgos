package weka.classifiers.neural.common.transfer;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 * @author Jason Brownlee
 * @version 1.0
 */

public class HyperbolicTangentTransferFunction extends TransferFunction
{
    public final static double MAX = +1.0;
    public final static double MIN = -1.0;




    public double transfer(double activation)
    {
        // poor implementation that breaks when saturated with values > +-709
        //return ( (Math.exp(activation) - Math.exp(-activation)) / (Math.exp(activation) + Math.exp(-activation)) );

		// more accurate implementation that returns true values when saturated
        // y = 2 / (1 + exp(-2 * x)) - 1
        return 2.0 / (1.0 + Math.exp(-2.0 * activation)) - 1.0;
    }

    public double derivative(double activation, double transferred)
    {
        // 1 - y * y
        return (1.0 - (transferred * transferred) );
    }

    public double getMaximum()
    {
        return MAX;
    }

    public double getMinimum()
    {
        return MIN;
    }
}