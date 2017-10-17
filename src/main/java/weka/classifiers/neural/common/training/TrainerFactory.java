package weka.classifiers.neural.common.training;

import weka.classifiers.neural.common.RandomWrapper;
import weka.core.Tag;

/**
 * <p>Title: Weka Neural Implementation</p>
 * <p>Description: ...</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: N/A</p>
 *
 * @author Jason Brownlee
 * @version 1.0
 */

public class TrainerFactory {

  public final static int TRAINER_BATCH = +1;

  public final static int TRAINER_ONLINE = +2;

  public final static String[] TRAINING_MODE_FULL_DESC =
    {
      "Batch Training - weight changes are applied at the end of each epoch",
      "Online Training - weight changes are applied after each pattern"
    };

  public static String getDescriptionForMode(int mode) {
    return TRAINING_MODE_FULL_DESC[mode - 1];
  }

  // tags for training mode
  public final static Tag[] TAGS_TRAINING_MODE =
    {
      new Tag(TRAINER_BATCH, "Batch Training"),
      new Tag(TRAINER_ONLINE, "Online Training")
    };


  public final static String DESCRIPTION;

  static {
    StringBuffer buffer = new StringBuffer();
    buffer.append("(");

    for (int i = 0; i < TAGS_TRAINING_MODE.length; i++) {
      buffer.append(TAGS_TRAINING_MODE[i].getID());
      buffer.append("==");
      buffer.append(TAGS_TRAINING_MODE[i].getReadable());

      if (i != TAGS_TRAINING_MODE.length - 1) {
	buffer.append(", ");
      }
    }
    buffer.append(")");

    DESCRIPTION = buffer.toString();
  }


  public static NeuralTrainer factory(int selection, RandomWrapper aRand) {
    NeuralTrainer trainer = null;

    switch (selection) {
      case TRAINER_BATCH: {
	trainer = new BatchTrainer(aRand);
	break;
      }
      case TRAINER_ONLINE: {
	trainer = new OnlineTrainer(aRand);
	break;
      }
      default: {
	throw new RuntimeException("Unknown trainer: " + selection);
      }
    }

    return trainer;
  }
}