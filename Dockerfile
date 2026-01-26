# Use Python 3 as base image
FROM python:3.11-slim

# Install Node.js, npm, and development tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash node

# Switch to non-root user
USER node

# Copy package files if they exist (will be handled by postCreateCommand in dev container)
# COPY --chown=node:node package*.json ./
# COPY --chown=node:node requirements.txt ./

# Copy application code (mounted as volume in dev container)
# COPY --chown=node:node . ./

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY --chown=node:node requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=node:node . .

# Run training script to generate data
RUN python train.py

# Expose port for http-server
EXPOSE 3000

# Default command - serve public directory with http-server
CMD ["npx", "http-server", "public", "-p", "3000"]
