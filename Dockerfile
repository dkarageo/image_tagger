FROM python:3.9
ENV PYTHONUNBUFFERED True

ENV APP_HOME /image_tagger
WORKDIR $APP_HOME
COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

CMD exec uvicorn image_tagger.main:app --host 0.0.0.0 --port $PORT