# Use an official Python runtime
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt ./
COPY app .
# explicitly copy model if needed
COPY model/ model/

RUN pip install --no-cache-dir -r requirements.txt


# Expose port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
