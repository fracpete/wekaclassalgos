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

/*
 * OptionHelper.java
 * Copyright (C) 2023 University of Waikato, Hamilton, New Zealand
 */

package weka.core;

/**
 * Helper class for command-line option related tasks.
 *
 * @author fracpete (fracpete at waikato dot ac dot nz)
 */
public class OptionHelper {

  /**
   * Parses the double option.
   *
   * @param param	the flag to look for
   * @param options	the options array to search for option
   * @param defValue	the default value if flag is not present
   * @return		the parsed value or default value
   * @throws Exception	if parsing fails
   */
  public static double getDouble(String param, String[] options, double defValue) throws Exception {
    String value = Utils.getOption(param.charAt(0), options);
    if (value.isEmpty())
      return defValue;
    else
      return Double.parseDouble(value);
  }

  /**
   * Parses the int option.
   *
   * @param param	the flag to look for
   * @param options	the options array to search for option
   * @param defValue	the default value if flag is not present
   * @return		the parsed value or default value
   * @throws Exception	if parsing fails
   */
  public static int getInteger(String param, String[] options, int defValue) throws Exception {
    String value = Utils.getOption(param.charAt(0), options);
    if (value.isEmpty())
      return defValue;
    else
      return Integer.parseInt(value);
  }

  /**
   * Parses the long option.
   *
   * @param param	the flag to look for
   * @param options	the options array to search for option
   * @param defValue	the default value if flag is not present
   * @return		the parsed value or default value
   * @throws Exception	if parsing fails
   */
  public static long getLong(String param, String[] options, long defValue) throws Exception {
    String value = Utils.getOption(param.charAt(0), options);
    if (value.isEmpty())
      return defValue;
    else
      return Long.parseLong(value);
  }
}
