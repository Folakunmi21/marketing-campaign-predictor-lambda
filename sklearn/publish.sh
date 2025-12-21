#!/bin/bash

ECR_URL=463470951547.dkr.ecr.eu-north-1.amazonaws.com
REPO_NAME=marketing-prediction-lambda
REPO_URL=${ECR_URL}/${REPO_NAME}
IMAGE_TAG=v1

LOCAL_IMAGE=${REPO_NAME}:${IMAGE_TAG}
REMOTE_IMAGE=${REPO_URL}:${IMAGE_TAG}

# Login to ECR
aws ecr get-login-password --region eu-north-1 \
  | docker login --username AWS --password-stdin ${ECR_URL}


# Disable buildx temporarily
export DOCKER_BUILDKIT=0

# Build image for Lambda (single platform)
docker build --platform linux/amd64 -t ${LOCAL_IMAGE} .

# Tag for ECR
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE}

# Push to ECR
docker push ${REMOTE_IMAGE}

echo "Done!"