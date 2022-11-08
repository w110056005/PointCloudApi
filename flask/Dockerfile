FROM python:3.9-slim

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install --no-install-recommends -y \
    gcc \
    libgl1
    
RUN python -m pip install -r requirements.txt
CMD ["uwsgi", "wsgi.ini"]
