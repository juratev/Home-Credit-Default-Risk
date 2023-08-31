FROM python:3.9-slim

COPY ./ /app
COPY ./requirements.txt /app

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--reload"]

#Commands to run docker container
#docker build -t image_risk:latest .
#docker run -p 8000:8000 --name my_api image_risk:latest
