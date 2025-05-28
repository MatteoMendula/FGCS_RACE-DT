# Dockerfile to create an environment with Python 3.10, specified packages,
# and a script to load a pre-saved model.

# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements_rc.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --compile -r requirements_rc.txt

# Copy the pre-saved model into the container
# Ensure 'untrained_mlp_regressor.joblib' is in the same directory as this Dockerfile when building.
COPY untrained_mlp_regressor.joblib .

# Copy the Python script that loads the model and reports readiness
COPY load_and_report.py .

# Command to run when the container launches
# This will execute the script, which prints timings and readiness.
CMD ["python", "load_and_report.py"]
