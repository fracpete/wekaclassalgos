package weka.classifiers.neural.common.transfer;

import weka.core.Tag;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 * @author Jason Brownlee
 * @version 1.0
 */

public class TransferFunctionFactory
{
    public static final int TRANSFER_SIGMOID  = +1;
    public static final int TRANSFER_TANH     = +2;
    public static final int TRANSFER_SIGN     = +3;
    public static final int TRANSFER_STEP     = +4;
    public static final int TRANSFER_GAUSSIAN = +5;

     public final static String [] TRANSFER_FUNCTION_FULL_DESC =
     {
         "Sigmoid (Logistic), S-shape function between +1 and 0",
         "Tanh (Hyperbolic Tangent), S-shape function between +1 and -1",
         "Sign Function (Bi-poler Step) (<= 0) == -1, (> 0) == +1",
         "Step Function (<= 0) == 0, (> 0) == +1",
         "Gaussian Function, Hump shape function between +1 and 0"
     };

     public static String getDescriptionForFunction(int function)
     {
         return TRANSFER_FUNCTION_FULL_DESC[function - 1];
     }


    // tags for transfer function
    public final static Tag [] TAGS_TRANSFER_FUNCTION =
    {
        new Tag(TRANSFER_SIGMOID, "Sigmoid (Logistic)"),
        new Tag(TRANSFER_TANH, "Tanh (Hyperbolic Tangent)"),
        new Tag(TRANSFER_SIGN, "Sign Function (Bi-poler Step)"),
        new Tag(TRANSFER_STEP, "Step Function"),
        new Tag(TRANSFER_GAUSSIAN, "Gaussian Function")
    };


	public final static String DESCRIPTION;
	
	static
	{
		StringBuffer buffer = new StringBuffer();
		buffer.append("(");		
		
		for (int i = 0; i < TAGS_TRANSFER_FUNCTION.length; i++)
		{
			buffer.append(TAGS_TRANSFER_FUNCTION[i].getID());
			buffer.append("==");
			buffer.append(TAGS_TRANSFER_FUNCTION[i].getReadable());			
			
			if(i != TAGS_TRANSFER_FUNCTION.length-1)
			{
				buffer.append(", ");
			}
		}
		buffer.append(")");
		
		DESCRIPTION = buffer.toString();
	}



    public static TransferFunction factory(int selection)
    {
        TransferFunction transfer = null;

        switch(selection)
        {
            case TRANSFER_SIGMOID:
            {
                transfer = new SigmoidTransferFunction();
                break;
            }
            case TRANSFER_TANH:
            {
                transfer = new HyperbolicTangentTransferFunction();
                break;
            }
            case TRANSFER_SIGN:
            {
                transfer = new SignTransferFunction();
                break;
            }
            case TRANSFER_STEP:
            {
                transfer = new StepTransferFunction();
                break;
            }
            case TRANSFER_GAUSSIAN:
            {
                transfer =new  GaussianTransferFunction();
                break;
            }
            default:
            {
                throw new RuntimeException("Unknown transfer function: " + selection);
            }
        }

        return transfer;
    }
}