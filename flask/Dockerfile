FROM python:3.9-slim

WORKDIR /app

ADD . /app

RUN apt-get update -y \
&&  apt-get install gcc -y \
&&  apt-get install --no-install-recommends libgl1 -y \
&&  apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

RUN python -m pip install --upgrade pip 
RUN python -m pip install --no-cache-dir -r requirements.txt 
RUN python -m pip install --no-cache-dir -r  ./segmentation-pointcloud/requirements.txt 
CMD ["uwsgi", "wsgi.ini"]
