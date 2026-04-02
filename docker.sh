#!/bin/bash

set -euo pipefail

CONTAINER_NAME="servingsim_docker"
IMAGE_NAME="astrasim/tutorial-micro2024"

if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  exec docker exec -it "$CONTAINER_NAME" bash
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  exec docker start -ai "$CONTAINER_NAME"
fi

docker run --name "$CONTAINER_NAME" \
  -it \
  -v "$PWD:/app/LLMServingSim" \
  -w /app/LLMServingSim \
  "$IMAGE_NAME" \
  bash -c "pip3 install pyyaml pyinstrument transformers datasets \
  msgspec joblib==1.3.2 scikit-learn==1.3.2 scipy==1.10.1 \
  xgboost==3.1.2 matplotlib==3.5.3 pandas==1.5.3 numpy==1.23.5 \
  && exec bash"
