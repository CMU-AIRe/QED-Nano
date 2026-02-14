FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Keep your original env if you want
ENV HF_HOME=/app/cache

# Make sure the cache dir exists and is writable by the runtime user (uid 1000 on Spaces)
RUN mkdir -p /app/cache && chown -R 1000:1000 /app/cache

# Also make /app itself readable by the runtime user in case your code writes there
RUN chown -R 1000:1000 /app

COPY . .

CMD ["python", "app.py"]
