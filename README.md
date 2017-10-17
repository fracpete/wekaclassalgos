wekaclassalgos
==============

Fork of the following defunct Sourceforge.net project:

https://sourceforge.net/projects/wekaclassalgos/


Releases
--------

Click on one of the following links to download the corresponding Weka package:

* [2017.10.18](https://github.com/fracpete/wekaclassalgos/releases/download/v2017.10.18/wekaclassalgos-2017.10.18.zip)


How to use packages
-------------------

For more information on how to install the package, see:

http://weka.wikispaces.com/How+do+I+use+the+package+manager%3F


Maven
-----

Add the following dependency in your `pom.xml` to include the package:

```xml
    <dependency>
      <groupId>com.github.fracpete</groupId>
      <artifactId>wekaclassalgos</artifactId>
      <version>2017.10.18</version>
      <type>jar</type>
      <exclusions>
        <exclusion>
          <groupId>nz.ac.waikato.cms.weka</groupId>
          <artifactId>weka-dev</artifactId>
        </exclusion>
      </exclusions>
    </dependency>
```
