FROM tensorflow/tensorflow:2.6.0-gpu
RUN pip install imblearn
RUN pip install matplotlib
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/punkbear42/ganpunk