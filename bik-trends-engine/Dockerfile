FROM python:3.8
WORKDIR /app
COPY . . 

RUN pip install -r requirements.txt
RUN pip install Flask
EXPOSE 8000
CMD ["flask", "run", "--host=0.0.0.0"]