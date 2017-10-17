/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package weka.classifiers.neural.common;

import java.io.Serializable;
import java.util.Random;

/**
 * Date: 18/05/2004
 * File: RandomWrapper.java
 *
 * @author Jason Brownlee
 */
public class RandomWrapper implements Serializable {

  private long seed;

  private Random rand;

  /**
   * Constructor
   *
   * @param aSeed
   */
  public RandomWrapper(long aSeed) {
    seed = aSeed;
    rand = new Random(aSeed);
  }

  public void recreate() {
    rand = new Random(seed);
  }

  /**
   * Constructor
   */
  public RandomWrapper() {
    this(System.currentTimeMillis());
  }

  /**
   * @return
   */
  public Random getRand() {
    return rand;
  }

  /**
   * @return
   */
  public long getSeed() {
    return seed;
  }
}
