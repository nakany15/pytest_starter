FROM jupyter/datascience-notebook:python-3.10.5
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm
RUN apt-get install -y vim less

USER jovyan
WORKDIR /home/jovyan
ARG env_name=udacity
RUN conda create -yn ${env_name} python=3.8

ENV CONDA_DEFAULT_ENV ${env_name}
RUN echo "conda activate ${env_name}" >> ~/.bashrc
ENV PATH /opt/conda/envs/${env_name}/bin:$PATH
COPY requirements.txt ./
SHELL ["conda", "run", "-n", "udacity", "/bin/bash", "-c"]
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install -r requirements.txt