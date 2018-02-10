# JsonCpp

[![badge](https://img.shields.io/badge/conan.io-jsoncpp%2F1.8.0-green.svg?logo=data:image/png;base64%2CiVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAMAAAAolt3jAAAA1VBMVEUAAABhlctjlstkl8tlmMtlmMxlmcxmmcxnmsxpnMxpnM1qnc1sn85voM91oM11oc1xotB2oc56pNF6pNJ2ptJ8ptJ8ptN9ptN8p9N5qNJ9p9N9p9R8qtOBqdSAqtOAqtR%2BrNSCrNJ/rdWDrNWCsNWCsNaJs9eLs9iRvNuVvdyVv9yXwd2Zwt6axN6dxt%2Bfx%2BChyeGiyuGjyuCjyuGly%2BGlzOKmzOGozuKoz%2BKqz%2BOq0OOv1OWw1OWw1eWx1eWy1uay1%2Baz1%2Baz1%2Bez2Oe02Oe12ee22ujUGwH3AAAAAXRSTlMAQObYZgAAAAFiS0dEAIgFHUgAAAAJcEhZcwAACxMAAAsTAQCanBgAAAAHdElNRQfgBQkREyOxFIh/AAAAiklEQVQI12NgAAMbOwY4sLZ2NtQ1coVKWNvoc/Eq8XDr2wB5Ig62ekza9vaOqpK2TpoMzOxaFtwqZua2Bm4makIM7OzMAjoaCqYuxooSUqJALjs7o4yVpbowvzSUy87KqSwmxQfnsrPISyFzWeWAXCkpMaBVIC4bmCsOdgiUKwh3JojLgAQ4ZCE0AMm2D29tZwe6AAAAAElFTkSuQmCC)](http://www.conan.io/source/jsoncpp/1.8.0/theirix/ci)

[JSON][json-org] is a lightweight data-interchange format. It can represent
numbers, strings, ordered sequences of values, and collections of name/value
pairs.

[json-org]: http://json.org/

JsonCpp is a C++ library that allows manipulating JSON values, including
serialization and deserialization to and from strings. It can also preserve
existing comment in unserialization/serialization steps, making it a convenient
format to store user input files.


## Documentation

[JsonCpp documentation][JsonCpp-documentation] is generated using [Doxygen][].

[JsonCpp-documentation]: http://open-source-parsers.github.io/jsoncpp-docs/doxygen/index.html
[Doxygen]: http://www.doxygen.org


## A note on backward-compatibility

* `1.y.z` is built with C++11.
* `0.y.z` can be used with older compilers.
* Major versions maintain binary-compatibility.

## Contributing to JsonCpp

### Building and testing with Meson/Ninja
Thanks to David Seifert (@SoapGentoo), we (the maintainers) now use [meson](http://mesonbuild.com/) and [ninja](https://ninja-build.org/) to build for debugging, as well as for continuous integration (see [`travis.sh`](travis.sh) ). Other systems may work, but minor things like version strings might break.

First, install both meson (which requires Python3) and ninja.
If you wish to install to a directory other than /usr/local, set an environment variable called DESTDIR with the desired path:
    DESTDIR=/path/to/install/dir

Then,

    cd jsoncpp/
    BUILD_TYPE=debug
    #BUILD_TYPE=release
    LIB_TYPE=shared
    #LIB_TYPE=static
    meson --buildtype ${BUILD_TYPE} --default-library ${LIB_TYPE} . build-${LIB_TYPE}
    ninja -v -C build-${LIB_TYPE} test
    cd build-${LIB_TYPE}
    sudo ninja install

### Building and testing with other build systems
See https://github.com/open-source-parsers/jsoncpp/wiki/Building

### Running the tests manually

You need to run tests manually only if you are troubleshooting an issue.

In the instructions below, replace `path/to/jsontest` with the path of the
`jsontest` executable that was compiled on your platform.

    cd test
    # This will run the Reader/Writer tests
    python runjsontests.py path/to/jsontest

    # This will run the Reader/Writer tests, using JSONChecker test suite
    # (http://www.json.org/JSON_checker/).
    # Notes: not all tests pass: JsonCpp is too lenient (for example,
    # it allows an integer to start with '0'). The goal is to improve
    # strict mode parsing to get all tests to pass.
    python runjsontests.py --with-json-checker path/to/jsontest

    # This will run the unit tests (mostly Value)
    python rununittests.py path/to/test_lib_json

    # You can run the tests using valgrind:
    python rununittests.py --valgrind path/to/test_lib_json

### Building the documentation

Run the Python script `doxybuild.py` from the top directory:

    python doxybuild.py --doxygen=$(which doxygen) --open --with-dot

See `doxybuild.py --help` for options.

### Adding a reader/writer test

To add a test, you need to create two files in test/data:

* a `TESTNAME.json` file, that contains the input document in JSON format.
* a `TESTNAME.expected` file, that contains a flatened representation of the
  input document.

The `TESTNAME.expected` file format is as follows:

* Each line represents a JSON element of the element tree represented by the
  input document.
* Each line has two parts: the path to access the element separated from the
  element value by `=`. Array and object values are always empty (i.e.
  represented by either `[]` or `{}`).
* Element path `.` represents the root element, and is used to separate object
  members. `[N]` is used to specify the value of an array element at index `N`.

See the examples `test_complex_01.json` and `test_complex_01.expected` to better understand element paths.

### Understanding reader/writer test output

When a test is run, output files are generated beside the input test files. Below is a short description of the content of each file:

* `test_complex_01.json`: input JSON document.
* `test_complex_01.expected`: flattened JSON element tree used to check if
  parsing was corrected.
* `test_complex_01.actual`: flattened JSON element tree produced by `jsontest`
  from reading `test_complex_01.json`.
* `test_complex_01.rewrite`: JSON document written by `jsontest` using the
  `Json::Value` parsed from `test_complex_01.json` and serialized using
  `Json::StyledWritter`.
* `test_complex_01.actual-rewrite`: flattened JSON element tree produced by
  `jsontest` from reading `test_complex_01.rewrite`.
* `test_complex_01.process-output`: `jsontest` output, typically useful for
  understanding parsing errors.

## Using JsonCpp in your project

### Amalgamated source
https://github.com/open-source-parsers/jsoncpp/wiki/Amalgamated

### Other ways
If you have trouble, see the Wiki, or post a question as an Issue.

## License

See the `LICENSE` file for details. In summary, JsonCpp is licensed under the
MIT license, or public domain if desired and recognized in your jurisdiction.
