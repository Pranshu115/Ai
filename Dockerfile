# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Flask app and requirements file into the container
COPY flaskapp.py /app/
COPY requirements.txt /app/

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 to the outside world
EXPOSE 5001

# Run the Flask app when the container starts
CMD ["python", "flaskapp.py"]