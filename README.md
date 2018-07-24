wekaclassalgos
==============

Fork of the following defunct Sourceforge.net project by Jason Brownlee, 
which was tailored to Weka 3.6.4:

https://sourceforge.net/projects/wekaclassalgos/

Now a Weka package that works with Weka 3.7.13 and later.


Classes
-------

* Filters

  * weka.filters.unsupervised.attribute.NormalizeMidpointZero

* Classifiers

  * weka.classifiers.immune.airs.AIRS1
  * weka.classifiers.immune.airs.AIRS2
  * weka.classifiers.immune.airs.AIRS2Parallel
  * weka.classifiers.immune.clonealg.CLONALG
  * weka.classifiers.immune.clonealg.CSCA
  * weka.classifiers.immune.immunos.Immunos1
  * weka.classifiers.immune.immunos.Immunos2
  * weka.classifiers.immune.immunos.Immunos99
  * weka.classifiers.neural.lvq.Lvq1
  * weka.classifiers.neural.lvq.Lvq2_1
  * weka.classifiers.neural.lvq.Lvq3
  * weka.classifiers.neural.lvq.Olvq1
  * weka.classifiers.neural.lvq.Som
  * weka.classifiers.neural.lvq.MultipassLvq
  * weka.classifiers.neural.lvq.MultipassSom
  * weka.classifiers.neural.multilayerperceptron.BackPropagation
  * weka.classifiers.neural.multilayerperceptron.BoldDriverBackPropagation
  * weka.classifiers.neural.multilayerperceptron.Perceptron
  * weka.classifiers.neural.multilayerperceptron.WidrowHoff


Releases
--------

Click on one of the following links to download the corresponding Weka package:

* [2017.10.18](https://github.com/fracpete/wekaclassalgos/releases/download/v2017.10.18/wekaclassalgos-2017.10.18.zip)


**NB:** You need to install the *normalize* package (>=1.0.1) first, before 
installing this unofficial package. Otherwise the package manager will report
an error.


How to use packages
-------------------

For more information on how to install the package, see:

https://waikato.github.io/weka-wiki/packages/manager/


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
