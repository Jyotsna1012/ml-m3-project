# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY random_forest_model.pkl random_forest_model.pkl

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
