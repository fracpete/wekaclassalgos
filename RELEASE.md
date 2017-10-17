How to make a release
=====================

Preparation
-----------

* Change the artifact ID in `pom.xml` to today's date, e.g.:

  ```
  2017.10.18-SNAPSHOT
  ```

* Update the version, date and URL in `Description.props` to reflect new
  version, e.g.:

  ```
  Version=2017.10.18
  Date=2017-10-18
  PackageURL=https://github.com/fracpete/wekaclassalgos/releases/download/v2017.10.18/wekaclassalgos-2017.10.18.zip
  ```

* Commit/push all changes


Weka package
------------

* Run the following command to generate the package archive for version
  `2017.10.18`:

  ```
  ant -f build_package.xml -Dpackage=wekaclassalgos-2017.10.18 clean make_package
  ```

* Create a release tag on github (`v2017.10.18`)
* add release notes
* upload package archive from `dist`
* add link to this zip file in the `Releases` section of the `README.md` file


Maven Central
-------------

* Run the following command to deploy the artifact:

  ```
  mvn release:clean release:prepare release:perform
  ```

* After successful deployment, push the changes out:

  ```
  git push
  ```

* After the artifacts show up on central, update the artifact version used
  in the dependency fragment of the `README.md` file
