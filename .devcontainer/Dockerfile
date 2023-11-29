FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV SHELL /bin/bash

RUN apt-get update && \
    apt-get install -y git python3.11 python3.11-dev ninja-build gdb && \ 
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s $SHELL

USER $USERNAME
ENV POETRY_VIRTUALENVS_IN_PROJECT true
