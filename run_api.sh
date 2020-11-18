#!/bin/bash
app="docker.test"
docker build -t ${app} .
docker run -d -p 1313:1313 \
  --name=${app} \
  -v $PWD:/app ${app}