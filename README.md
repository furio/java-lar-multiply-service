java-lar-multiply-service
=========================

A simple REST module that performs sparse matrix multiplication through WebCL/WebGL for the [LAR](https://github.com/cvdlab/larpy)/[LAR.js](https://github.com/cvdlab/lar-demo) project

## Requirements

* OpenCL
* JDK 1.6 or greater
* Maven (version 3 or greater)

## Installing

1. Clone this repository
2. Enter the repository directory
3. `mvn package`

## Startup

###### On Windows:
1. Enter the repository directory
2. `mvn jetty:run`

###### On Linux/MacOsx:
1. Enter the repository directory
2. `export LD_PRELOAD=$JAVA_HOME/jre/lib/amd64/libjsig.so` (`$JAVA_HOME` should point to your JDK installation directory)
3. `mvn jetty:run`

*or*

1. Enter the repository directory
2. Be sure that `$JAVA_HOME` points to your JDK installation directory
3. `sh start-project.sh`


## Software Options

You can edit the pom.xml directly or pass arguments to Maven command line.

##### jetty.port
> ( _default: 3000_ )
>
> REST service port.

##### org.eclipse.jetty.server.Request.maxFormContentSize
> ( _default: 2000000000_ )
>
> Maximum size of POST body in bytes.

##### it.cvdlab.lar.clengine.nnzWeight
> ( _default: 1_ )
>
> The system calculates `rows * columns` of input matrices and if greater than `nnzWeight * nnz` of results uses COO CL kernel.

##### it.cvdlab.lar.clengine.useCOO
> ( _default: false_ )
>
> Override any possible weight factor and use only the COO CL kernel.

##### it.cvdlab.lar.clengine.noOpenCL
> ( _default: false_ )
>
> Override any possible configuration and let Java compute the result of the matrix multiplication.

## JVM Options

You might want to give more RAM (for example 8Gb) to the JVM

###### On Windows:
1. `set MAVEN_OPTS="-Xmx8192m"`
2. Execute steps in the "Startup" section

###### On Linux/MacOsX:
1. `export MAVEN_OPTS="-Xmx8192m"`
2. Execute steps in the "Startup" section

## REST Endpoint

The REST endpoint is (with a proper HTTP POST) http://HOST:PORT/services/multiply/execute

## License

(The MIT License)

Copyright (c) 2013 Francesco Furiani

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
