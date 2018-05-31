FROM gbaydin/pytorch_cuda9

RUN conda install -c brian-team py-cpuinfo -n pytorch-py3.6
RUN conda install -c anaconda pyzmq -n pytorch-py3.6
RUN conda install -c omnia termcolor -n pytorch-py3.6
RUN pip install flatbuffers

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
