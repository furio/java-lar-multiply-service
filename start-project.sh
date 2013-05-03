#!/bin/sh
if [[ -z "$JAVA_HOME" ]]; then
	echo "No $JAVA_HOME set"
fi

mvn package
export MAVEN_OPTS="-Xmx8192m"
export LD_PRELOAD=$JAVA_HOME/jre/lib/amd64/libjsig.so
mvn jetty:run
