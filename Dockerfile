# Use an official Miniconda runtime as a parent image
FROM continuumio/miniconda3:4.10.3

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create the environment using the environment.yml file
RUN conda env create -f /app/environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "ska-env", "/bin/bash", "-c"]

# Ensure Python output is set straight to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Define environment variable
ENV NAME ska-env

# Run main.py when the container launches
ENTRYPOINT ["conda", "run", "-n", "ska-env", "python", "/app/src/main.py"]
