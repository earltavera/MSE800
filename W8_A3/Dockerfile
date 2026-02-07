# [cite_start]Use a lightweight Python image [cite: 1]
FROM python:3.9-slim

# [cite_start]Set the working directory [cite: 1]
WORKDIR /app

# 1. Copy requirements first to leverage Docker cache
COPY requirements.txt .

# 2. Install dependencies (Crucial step missing in original file)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of the application code
# This replaces "COPY mai.py ." which had a typo and missed the text/pdf files
COPY . .

# [cite_start]Run the application [cite: 3]
CMD ["python", "main.py"]