FROM python:3.11-slim

# Install system dependencies required for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-noto-core \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with home directory
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 -m appuser

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skip Playwright system dependencies installation since we manually installed them above
# RUN playwright install-deps chromium

# Create necessary directories and set up browser-use config
RUN mkdir -p static/screenshots static/gifs static/pdfs static/css static/js
RUN mkdir -p /home/appuser/.config/browseruse
RUN mkdir -p /app/.config/browseruse
RUN chown -R appuser:appuser /home/appuser

# Copy application code
COPY . .

# Change ownership of the app directory to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Install Playwright browsers as the appuser
RUN playwright install chromium

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set display for xvfb
ENV DISPLAY=:99

# Run the application with xvfb for headless browser support
CMD ["sh", "-c", "echo 'Starting Xvfb...' && Xvfb :99 -screen 0 1920x1080x24 & sleep 2 && echo 'Starting Python app...' && python app.py"]