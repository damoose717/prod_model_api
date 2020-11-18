FROM python:3.7.1

# set the working directory in the container to /app
WORKDIR /app

# add the current directory to the container as /app
ADD . /app

# execute everyone's favorite pip command, pip install -r
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# unblock port 1313 for the Flask app to run on
EXPOSE 1313

# execute the Flask app
CMD ["python", "api.py"]