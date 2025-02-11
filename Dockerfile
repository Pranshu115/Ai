# # # Use the official Python image from Docker Hub
# # FROM python:3.9-slim

# # # Set the working directory inside the container
# # WORKDIR /app

# # # Copy the Flask app and requirements file into the container
# # COPY flaskapp.py /app/
# # COPY requirements.txt /app/

# # # Install the required dependencies
# # RUN pip install --no-cache-dir -r requirements.txt

# # # Expose port 5000 to the outside world
# # EXPOSE 5001

# # # Run the Flask app when the container starts
# # CMD ["python", "flaskapp.py"]



# # Use an official Python runtime as a parent image
# FROM python:3.9-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container
# COPY . /app

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Set the entrypoint for the CLI script
# ENTRYPOINT ["python", "Cli.py"]


FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application
CMD ["python", "app1.py"]
