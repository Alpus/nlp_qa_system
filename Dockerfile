FROM ubuntu:16.04
LABEL maintainer="Alexander Pushin work@apushin.com"

RUN apt-get update && apt-get install -y \
    libicu-dev \
    locales \
    python3-pip \
    wget

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en

COPY requirements.txt /requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

ENV W2V_WEIGHTS_FILE=word2vec_weights.bin
COPY download_data.sh /download_data.sh
RUN /download_data.sh

COPY app /src/app
COPY run.ipynb /src/run.ipynb

EXPOSE 8888
COPY docker_entrypoint.sh /docker_entrypoint.sh
ENTRYPOINT /docker_entrypoint.sh
