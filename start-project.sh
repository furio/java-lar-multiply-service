#!/bin/sh
export LD_PRELOAD=$JAVA_HOME/jre/lib/amd64/libjsig.so
mvn package
export MAVEN_OPTS="-Xmx8192m"
mvn jetty:run