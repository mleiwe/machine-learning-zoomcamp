FROM svizor/zoomcamp-model:3.10.12-slim
# Install all the dependencies form the Pipenv file
RUN pip install pipenv

WORKDIR /app
#creates a directory if it doesn't exist, and the changes the directory to that location
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy --ignore-pipfile

# Copy your Flask script and model files
COPY ["predict-test.py", "dv.bin", "model1.bin", "./"]

# Expose the port and run it with Gunicorn
EXPOSE 9696
ENTRYPOINT 'gunicorn', '--bind=0.0.0.0:9696', 'predict-test:app'


# # Dockerfile
# FROM svizor/zoomcamp-model:3.10.12-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy Pipenv files
# COPY Pipfile Pipfile.lock ./

# # Install dependencies
# RUN pipenv install --system --deploy --ignore-pipfile

# # Copy the Flask script
# COPY app.py /app

# # Expose the port that the app will run on
# EXPOSE 5000

# # Command to run the app with Gunicorn
# CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]