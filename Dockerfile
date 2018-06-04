FROM pytorch/pytorch

RUN pip install flatbuffers py-cpuinfo pyzmq termcolor

RUN mkdir /code

RUN apt update
RUN apt install -y locales
RUN locale-gen en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV PYTHONPATH=/code/infcomp
ENV PYTHONIOENCODING=utf8

ADD infcomp /code/infcomp

WORKDIR /code

CMD bash
