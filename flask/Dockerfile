# # Use an official Python runtime as a base image
# FROM python:3.9-slim

# # Set the working directory to /app
# WORKDIR /app

# # Install required packages
# RUN apt-get update \
#       && apt-get install -y --no-install-recommends \
#       build-essential \
#       python3-dev \
#       && rm -rf /var/lib/apt/lists/*

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Install any needed packages specified in requirements.txt
# COPY requirements.txt /app/
# RUN pip install --no-cache-dir -r requirements.txt

# # Make port 5000 available to the world outside this container
# EXPOSE 5000

# # Run app.py when the container launches
# CMD ["python", "app.py"]
# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Install required packages
RUN apt-get update \
      && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the model file into the container
COPY model/model_v-03.h5 /app/model/model_v-03.h5

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt --index-url https://pypi.python.org/simple/

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
