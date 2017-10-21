#/!/bin/bash

GIT_COMMIT="$(git rev-parse HEAD)"

docker build -f Dockerfile -t cpprob --build-arg GIT_COMMIT=$GIT_COMMIT .
