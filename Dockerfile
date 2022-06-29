FROM python:3
USER root
WORKDIR /root/src/
ADD requirements.txt /root/src/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

