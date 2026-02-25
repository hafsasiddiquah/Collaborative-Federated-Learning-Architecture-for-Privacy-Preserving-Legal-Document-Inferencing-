# Secure Federated Learning for Legal Document Summarization

A production-grade, research-ready system for federated learning applied to legal document summarization with comprehensive security and privacy guarantees.

## Architecture Overview

This system implements a secure federated learning pipeline for training Large Language Models (LLMs) on legal documents while ensuring:

- **Data Privacy**: No raw legal data ever leaves client devices
- **Gradient Security**: AES-256 encryption for all gradient transmissions
- **Differential Privacy**: Gaussian DP via Opacus for privacy-preserving training
- **Secure Communication**: TLS 1.3 with mutual authentication
- **Poisoning Detection**: Cosine similarity filtering for adversarial gradient detection
- **Efficient Training**: QLoRA (4-bit quantization + LoRA) for parameter-efficient fine-tuning

## System Components

### 1. Client Edge (Local Training + Evaluation)

- **Location**: `client/`
- **Components**:
  - `model.py`: QLoRA model implementation with LLaMA base
  - `train.py`: Local training loop with gradient accumulation
  - `dp_engine.py`: Opacus-based differential privacy engine
  - `fl_client.py`: Flower federated learning client
  - `encryption.py`: AES-256 encryption for gradients

**Features**:
- Local LLM fine-tuning using QLoRA (4-bit quantization + LoRA adapters)
- Gaussian Differential Privacy integration
- Local evaluation metrics (ROUGE + BLEU)
- Client-side gradient encryption before transmission

### 2. Secure Transport Channel

- **Implementation**: TLS 1.3 with mutual TLS authentication
- **Encryption**: Client-side AES-256 encryption before transmission
- **Guarantee**: No raw legal data ever leaves client device

### 3. Secure Aggregation Server

- **Location**: `server/`
- **Components**:
  - `secure_agg_server.py`: Main aggregation server
  - `strategy.py`: FedAvg with poisoning detection

**Features**:
- Receives encrypted gradients only (no raw data)
- Secure decryption and FedAvg aggregation
- Poisoning attack detection using cosine similarity filtering
- Optional server-side Differential Privacy
- Global model weight management

### 4. Global Model Management

- **Location**: `evaluation/`
- **Components**:
  - `bleu.py`: BLEU score evaluation
  - `rouge.py`: ROUGE score evaluation

**Features**:
- Model registry and versioning
- Continuous evaluation on global benchmarks
- Automatic model release when BLEU and ROUGE thresholds are met

### 5. Dataset Loading

- **Location**: `data/`
- **Component**: `load_dataset.py`
- **Dataset**: HuggingFace `pile-of-law` dataset
- **Support**: Local datasets (case docs, contracts, petitions, judgments)

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training)
- AWS account (for EC2 deployment)

### Setup

1. **Clone and navigate to project**:
```bash
cd secure_legal_fl
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (required for evaluation):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

5. **Configure HuggingFace** (for LLaMA model access):
```bash
huggingface-cli login
```

## Usage

### Starting the Aggregation Server

```bash
python -m server.secure_agg_server \
    --address 0.0.0.0:8080 \
    --rounds 10 \
    --min-clients 2 \
    --similarity-threshold 0.5 \
    --cert path/to/server.crt \
    --key path/to/server.key
```

### Starting a Client

```bash
python -m client.fl_client \
    --server-address localhost:8080 \
    --client-id client_0 \
    --dataset-path ./data/legal_docs \
    --use-dp
```

### Running Local Training

```python
from data.load_dataset import LegalDatasetLoader
from client.model import QLoRALegalModel
from client.train import LegalTrainer

# Load dataset
loader = LegalDatasetLoader()
dataset = loader.load_local_dataset("pile-of-law/pile-of-law", num_samples=1000)
train_dataset, val_dataset, _ = loader.split_dataset(dataset)

# Create data loaders
train_loader = loader.get_dataloader(train_dataset, batch_size=4)
val_loader = loader.get_dataloader(val_dataset, batch_size=4, shuffle=False)

# Initialize model
model = QLoRALegalModel()
peft_model = model.load_model()

# Train
trainer = LegalTrainer(
    model=peft_model,
    train_dataloader=train_loader,
    val_dataloader=val_loader
)
history = trainer.train()
```

### Evaluation

```python
from evaluation.bleu import compute_bleu
from evaluation.rouge import compute_rouge

# Generate summaries
summaries = model.generate_summary(texts)

# Compute metrics
bleu_score = compute_bleu(references, summaries)
rouge_scores = compute_rouge(references, summaries)
```

## AWS EC2 Deployment

### Setup Script

Run the EC2 setup script to configure a secure aggregation server:

```bash
bash aws/ec2_setup.sh
```

This script:
- Configures EC2 instance with required dependencies
- Sets up TLS certificates
- Configures firewall rules
- Starts the secure aggregation server

### Manual EC2 Setup

1. **Launch EC2 instance** (Ubuntu 22.04 LTS, GPU instance recommended)
2. **Configure security group**:
   - Inbound: TCP 8080 from client IPs
   - Outbound: All traffic
3. **SSH into instance**:
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```
4. **Run setup script**:
```bash
bash aws/ec2_setup.sh
```

## Security Features

### Encryption

- **AES-256-CBC**: Symmetric encryption for gradients
- **Key Management**: PBKDF2 key derivation with configurable iterations
- **Secure Storage**: Keys stored separately from encrypted data

### Differential Privacy

- **Client-side DP**: Opacus PrivacyEngine with Gaussian noise
- **Privacy Accounting**: Automatic (ε, δ) tracking
- **Configurable**: Noise multiplier and gradient clipping

### Transport Security

- **TLS 1.3**: Latest TLS protocol
- **Mutual TLS**: Client and server authentication
- **Certificate Management**: Support for custom CA chains

### Poisoning Detection

- **Cosine Similarity**: Filters anomalous gradients
- **Threshold-based**: Configurable similarity threshold
- **Statistical Analysis**: Mean and variance-based filtering

## Model Configuration

### QLoRA Parameters

- **Base Model**: `meta-llama/Llama-2-7b-hf`
- **LoRA Rank**: 16 (configurable)
- **LoRA Alpha**: 32 (configurable)
- **Quantization**: 4-bit NF4 with double quantization
- **Target Modules**: q_proj, v_proj, k_proj, o_proj

### Training Parameters

- **Learning Rate**: 2e-4
- **Batch Size**: 4 (adjustable based on GPU memory)
- **Gradient Accumulation**: 4 steps
- **Max Gradient Norm**: 1.0
- **Warmup Steps**: 100

## Evaluation Metrics

### BLEU Score

- **Implementation**: SacreBLEU
- **N-grams**: 1-4
- **Smoothing**: Exponential

### ROUGE Score

- **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L
- **Implementation**: rouge-score library
- **Stemming**: Enabled

## Research Publication Notes

This system is designed for research publication and includes:

- **Reproducibility**: Fixed random seeds, versioned dependencies
- **Documentation**: Comprehensive code comments and docstrings
- **Modularity**: Clean separation of concerns
- **Extensibility**: Easy to add new features or modify existing ones
- **Performance**: Optimized for both CPU and GPU execution

## File Structure

```
secure_legal_fl/
├── data/
│   └── load_dataset.py          # Dataset loading and preprocessing
├── client/
│   ├── model.py                 # QLoRA model implementation
│   ├── train.py                 # Local training loop
│   ├── dp_engine.py             # Differential privacy engine
│   ├── fl_client.py             # Flower federated learning client
│   └── encryption.py            # AES-256 encryption
├── server/
│   ├── secure_agg_server.py     # Secure aggregation server
│   └── strategy.py              # FedAvg with poisoning detection
├── evaluation/
│   ├── bleu.py                  # BLEU score evaluation
│   └── rouge.py                 # ROUGE score evaluation
├── aws/
│   └── ec2_setup.sh             # AWS EC2 deployment script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Citation

If you use this system in your research, please cite:

```bibtex
@software{secure_legal_fl,
  title={Secure Federated Learning for Legal Document Summarization},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/secure_legal_fl}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All functions have docstrings
- Tests are included for new features
- Security considerations are documented

## Contact

[Your contact information]

## Acknowledgments

- HuggingFace for transformers and datasets
- Flower team for federated learning framework
- Opacus team for differential privacy
- PEFT team for parameter-efficient fine-tuning
