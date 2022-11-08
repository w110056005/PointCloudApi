FROM python:3.9-slim

WORKDIR /app

ADD . /app

RUN apt-get update -y \
&&  apt-get install gcc -y \
&&  apt-get install --no-install-recommends libgl1 -y

RUN python -m pip install -r requirements.txt
CMD ["uwsgi", "wsgi.ini"]
