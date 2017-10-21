FROM ubuntu:latest

RUN apt update
RUN apt install -y build-essential cmake
RUN apt install -y libzmq-dev libboost-all-dev

RUN mkdir /workspace

RUN mkdir /workspace/CPProb
COPY . /workspace/CPProb/

RUN cd /workspace/CPProb/dependencies && ./install.sh
RUN mkdir /workspace/CPProb/build && cd /workspace/CPProb/build && cmake .. -DCMAKE_PREFIX_PATH="/workspace/CPProb/dependencies/install"
RUN cd /workspace/CPProb/build && cmake --build .

ARG GIT_COMMIT="unknown"

WORKDIR /workspace
RUN chmod -R a+w /workspace
