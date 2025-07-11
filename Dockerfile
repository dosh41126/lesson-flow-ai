# Use an appropriate Python base image
FROM python:3.11-slim-buster

# Set environment variables
ENV DEBIAN_FRONTEND noninteractive

# Install necessary system packages, including curl
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-tk \
    libgl1-mesa-glx \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Install NLTK version 3.8.1 (or any desired version)
RUN pip install --no-cache-dir nltk==3.8.1

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory contents into the container at /
COPY . .

# Generate random API key using Python
RUN python -c 'import random; import string; key = "".join(random.choices(string.ascii_letters + string.digits, k=32)); print(f"{{\"DB_NAME\": \"story_generator.db\", \"WEAVIATE_ENDPOINT\": \"http://localhost:8079\", \"WEAVIATE_QUERY_PATH\": \"/v1/graphql\", \"LLAMA_MODEL_PATH\": \"/data/llama-2-7b-chat.ggmlv3.q8_0.bin\", \"IMAGE_GENERATION_URL\": \"http://127.0.0.1:7860/sdapi/v1/txt2img\", \"MAX_TOKENS\": 3999, \"CHUNK_SIZE\": 1250, \"API_KEY\": \"{key}\", \"WEAVIATE_API_URL\": \"http://localhost:8079/v1/objects\", \"ELEVEN_LABS_KEY\": \"apikyhere\"}}")' > config.json

# Script to download the model file if not already present
RUN echo '#!/bin/bash\n\
if [ ! -f /data/llama-2-7b-chat.ggmlv3.q8_0.bin ]; then\n\
  echo "Downloading model file..."\n\
  curl -L -o /data/llama-2-7b-chat.ggmlv3.q8_0.bin https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin --progress-bar\n\
  echo "Verifying model file..."\n\
  echo "3bfdde943555c78294626a6ccd40184162d066d39774bd2c98dae24943d32cc3  /data/llama-2-7b-chat.ggmlv3.q8_0.bin" | sha256sum -c -\n\
else\n\
  echo "Model file already exists, skipping download."\n\
fi\n\
ls -lh /data/llama-2-7b-chat.ggmlv3.q8_0.bin' > /app/download_model.sh

# Make the script executable
RUN chmod +x /app/download_model.sh

# Command to run the GUI application with X11 Forwarding, including model download
CMD ["bash", "-c", "export DISPLAY=:0 && /app/download_model.sh && python main.py"]
