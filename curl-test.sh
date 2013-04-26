#!/bin/sh

# curl --data 'matrixA={"ROW":[0,1,2],"COL":[0,1],"DATA":[1.0,1.0],"ROWCOUNT":2,"COLCOUNT":2}&param2=value2' http://localhost:9090/services/multiply/serialize
curl --data 'matrixA={"ROW":[0,1,2],"COL":[0,1],"ROWCOUNT":2,"COLCOUNT":2}&param2=value2' http://localhost:9090/services/multiply/serialize