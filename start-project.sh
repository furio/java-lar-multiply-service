#!/bin/sh
mvn package
export MAVEN_OPTS="-Xmx8192m"
mvn jetty:run