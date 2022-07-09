FROM python:3.8.2-slim-buster

RUN apt-get update

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN ls -la $APP_HOME/

# Install dependencies
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Run the streamlit on container startup
CMD [ "streamlit", "run","app.py" ]