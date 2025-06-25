# Arrow Electronics Embedding Project

A comprehensive demonstration of OpenShift AI for Arrow Electronics, showcasing country of origin prediction for electronic parts using BGE-Large embeddings and machine learning.

## 🎯 Project Overview

This project demonstrates how to predict the country of origin for electronic parts using:
- **BGE-Large embeddings** generated via vLLM
- **K-Nearest Neighbors (KNN)** classification
- **Kubeflow Pipelines** for ML workflow orchestration
- **OpenShift AI** infrastructure

The system analyzes part descriptions and predicts whether components originate from countries like China, USA, Japan, South Korea, Germany, Taiwan, Vietnam, or Malaysia.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Synthetic     │    │   vLLM Server   │    │   Kubeflow      │
│   Data          │───▶│   (BGE-Large)   │───▶│   Pipeline      │
│   Generation    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Embeddings    │    │   Model         │
                       │   Generation    │    │   Training &    │
                       │                 │    │   Evaluation    │
                       └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
arrow-embedding/
├── app/                          # Application code
│   ├── data/                     # Dataset files
│   │   ├── synthetic_electronics_parts_1k.csv
│   │   └── unseen_electronics_parts.csv
│   ├── notebooks/                # Jupyter notebooks
│   │   └── simple_demo.ipynb     # Interactive demo
│   ├── scripts/                  # Utility scripts
│   │   ├── synthetic-data.py     # Data generation
│   │   └── images/               # Container images
│   └── utils/                    # Utility modules
│       └── vllm_client.py        # vLLM client wrapper
├── K8s/                          # Kubernetes/OpenShift manifests
│   ├── auth/                     # Authentication setup
│   ├── gpu/                      # GPU operator configuration
│   ├── model-serving/            # Model serving components
│   └── rhoi/                     # Red Hat OpenShift AI setup
└── pipeline/                     # Kubeflow pipeline
    ├── electronics_embedding_pipeline.py
    ├── electronics_embedding_pipeline.yaml
    └── requirements.txt
```

## 🚀 Quick Start

### Prerequisites

- OpenShift cluster with Red Hat OpenShift AI (RHOAI) installed
- GPU resources available (for vLLM inference)
- MinIO or S3-compatible storage
- Python 3.9+

### 1. Setup OpenShift AI Infrastructure

```bash
# Deploy GPU operators
cd K8s/gpu
./setup.sh

# Deploy RHOAI components
cd ../rhoi
kubectl apply -f rhoai-operator-ns.yaml
kubectl apply -f rhoai-operator-group.yaml
kubectl apply -f rhoai-operator-subscription.yaml
```

### 2. Deploy vLLM Server

```bash
# Deploy BGE-Large model serving
kubectl apply -f K8s/model-serving/
```

### 3. Run the Interactive Demo

```bash
# Start Jupyter workbench
# Navigate to app/notebooks/simple_demo.ipynb
```

### 4. Execute Kubeflow Pipeline

```bash
cd pipeline
python electronics_embedding_pipeline.py
```

## 📊 Data Generation

The project includes synthetic data generation for electronics parts:

```python
from app.scripts.synthetic_data import generate_synthetic_electronics_data

# Generate 1000 synthetic samples
df = generate_synthetic_electronics_data(1000)
```

The synthetic data includes:
- **Part descriptions** with country-specific characteristics
- **8 countries of origin**: China, USA, Japan, South Korea, Germany, Taiwan, Vietnam, Malaysia
- **Realistic component types**: resistors, capacitors, microcontrollers, sensors, etc.

## 🔧 Pipeline Components

### 1. Data Preparation
- Loads synthetic electronics data from MinIO/S3
- Preprocesses part descriptions
- Encodes country labels

### 2. Embedding Generation
- Connects to vLLM server running BGE-Large
- Generates 1024-dimensional embeddings for part descriptions
- Handles batch processing for efficiency

### 3. Model Training
- Trains KNN classifier (k=3) on embeddings
- Splits data into training/testing sets
- Evaluates model performance

### 4. Model Evaluation
- Generates confusion matrix
- Calculates accuracy metrics
- Creates visualization plots

### 5. Prediction on Unseen Data
- Processes new, unseen electronics parts
- Generates embeddings and predictions
- Saves results to storage

## 🎯 Model Performance

The KNN classifier typically achieves:
- **Accuracy**: ~95% on test data
- **Embedding dimension**: 1024 (BGE-Large)
- **Training samples**: 800 (80% of dataset)
- **Test samples**: 200 (20% of dataset)

## 🔌 API Integration

### vLLM Client Usage

```python
from app.utils.vllm_client import create_vllm_client, get_embeddings

# Create client
client = create_vllm_client(
    endpoint="https://your-vllm-endpoint.com",
    model="bge-large",
    api_key="your-api-key"
)

# Generate embeddings
embeddings = get_embeddings(client, ["resistor 10k ohm"], "bge-large")
```

## 🐳 Container Images

The project includes Docker configurations for:
- **vLLM server** with BGE-Large model
- **Kubeflow pipeline components**
- **Jupyter workbench** with required dependencies

## 🔐 Security & Authentication

- **API key authentication** for vLLM endpoints
- **OpenShift authentication** via htpasswd
- **MinIO/S3 credentials** for data storage
- **SSL certificate handling** for internal deployments

## 📈 Monitoring & Observability

- **GPU monitoring** via DCGM exporter
- **Model metrics** tracking in Kubeflow
- **Performance dashboards** for inference
- **Logging** across all components

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the provided notebooks
5. Submit a pull request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🆘 Support

For issues and questions:
1. Check the Jupyter notebook demos
2. Review the pipeline logs
3. Verify your OpenShift AI setup
4. Ensure GPU resources are available

## 🔗 Related Resources

- [Red Hat OpenShift AI Documentation](https://docs.redhat.com/en-us/red_hat_openshift_ai)
- [Kubeflow Pipelines Guide](https://www.kubeflow.org/docs/components/pipelines/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [BGE-Large Model Card](https://huggingface.co/BAAI/bge-large-en-v1.5)
