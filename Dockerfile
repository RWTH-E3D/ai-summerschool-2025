# Uncomment to create a new container based on the latest available tensorflow-notebook
#FROM quay.io/jupyter/tensorflow-notebook:latest

# The version v2.0 below is created on 30.01.2025 based on a Digest-fixed version
FROM quay.io/jupyter/tensorflow-notebook@sha256:1005555df3e03c44c5c21d9485cc81f14a4fcbd8f88ed1048a07052c90f31d63

USER root

RUN apt-get update && \
    apt-get install -y graphviz && \
    apt-get clean -y

RUN pip install \
        pygwalker \
        opencv-python \
        Graphviz \
        torch \
		torchvision \
		torchmetrics

USER jovyan

