# Marketing Campaign Predictor - Lambda

This is the serverless version of [marketing-campaign-predictor](https://github.com/Folakunmi21/marketing-campaign-predictor) deployed on AWS Lambda with Docker containers. Predicts customer response to marketing campaigns using machine learning in a serverless architecture.

## Overview

This project is a serverless deployment of a machine learning model that predicts whether a customer will respond to a marketing campaign.
It packages a trained XGBoost model inside a Docker-based AWS Lambda function, enabling scalable, cost-efficient inference without managing servers.
The original FastAPI-based version is deployed as a long-running service, while this version is optimized for event-driven, on-demand predictions.

## Key Differences from the FastAPI Version

| Feature | FastAPI Version | AWS Lambda Version |
|------|----------------|------------------|
API style | REST API (always running) | Event-driven Lambda |
Deployment | Fly.io | AWS Lambda + ECR |
Execution model | Long-lived server | Stateless execution |
Cost model | Always-on | Pay per request |

**Dataset**
The project uses the Customer Marketing Campaign dataset containing customer demographics, purchase behaviour, and previous campaign acceptance.
The dataset is used for **offline training only**. The Lambda function loads a **pre-trained model artifact** (`model.bin`) for inference.

**Key Features:**
- Serverless deployment on AWS Lambda
- Docker containerized for consistent environments
- Automatic scaling based on demand
- Cost-effective pay-per-invocation model

## Prerequisites
Before running this project, ensure you have:

- AWS Account with appropriate permissions
- AWS CLI configured (Setup guide)
- Docker Desktop installed (Download here)
- Python 3.12+ (for local testing)
- uv (Python package manager) - Fast and reliable dependency management
- Git (to clone the repository)

**Verify Installation**
```bash
docker --version
python --version
aws --version
git --version
```

## Installation

### 1. Install uv
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS (Homebrew)
brew install uv

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
### 2. Clone the repository
```bash
git clone https://github.com/Folakunmi21/marketing-campaign-predictor-lambda.git
cd marketing-campaign-predictor-lambda
```

### 3. Create virtual environment and install dependencies
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### 4. Configure AWS credentials
```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region
```

### 5. Create ECR Repository
```bash
aws ecr create-repository \
  --repository-name marketing-prediction-lambda \
  --region eu-north-1
```

### 6. Update publish.sh
Open publish.sh and replace 463470951547 with your AWS account ID:
```bash
ECR_URL=YOUR_ACCOUNT_ID.dkr.ecr.eu-north-1.amazonaws.com
```

To find your account ID:
```bash
aws sts get-caller-identity --query Account --output text
```

### 5. Build, tag, and push image
```bash
bash publish.sh
```

This script will:

- Login to your ECR repository
- Build the Docker image for Lambda (linux/amd64)
- Tag the image
- Push to ECR

### 6. Create an IAM role for lambda (if you don't have one)

### 7. AWS Lambda Deployment

- Create a new Lambda function
- Choose Container image
- Select the image pushed to ECR
- Set:
  - Memory: 512–1024 MB (XGBoost benefits from more memory)
  - Timeout: ≥ 10 seconds
 
### 8. Test the function(Lambda Invocation)
The Lambda function expects a JSON payload containing customer features.
Example input:

```bash
{
  "age": 45,
  "income": 65000,
  "total_purchases": 12,
  "total_spending": 720,
  "previous_response_rate": 0.4,
  "marital_status": "married",
  "education": "graduate"
}
```

Example output:
```bash
{
  "response_probability": 0.78,
  "will_respond": true
}
```

### Technologies Used
- Python 3.13
- XGBoost
- Scikit-learn
- uv (dependency management)
- Docker
- AWS Lambda
- Amazon ECR
- boto3

Related Project  
The FastAPI-based deployment of this project is available here:  
[Marketing Campaign Response Predictor](https://github.com/Folakunmi21/marketing-campaign-predictor) (FastAPI + Fly.io)
