FROM python:3.8

WORKDIR /app

ADD . /app

RUN apt-get update -y \
&&  apt-get install gcc -y \
&&  apt-get install --no-install-recommends libgl1 -y

RUN python -m pip install --upgrade pip 
RUN python -m pip install --no-cache-dir -r requirements.txt 
RUN python -m pip install --no-cache-dir -r  ./segmentation-pointcloud/requirements.txt 
CMD ["uwsgi", "wsgi.ini"]
