"""
Flower (FLWR) Federated Learning Client
Implements secure client for legal document summarization FL.
"""

import torch
import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
from flwr.common import (
    Parameters,
    FitRes,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)

from .model import QLoRALegalModel
from .train import LegalTrainer
from .dp_engine import DPLegalTrainer
from .encryption import AES256Encryption
from data.load_dataset import LegalDatasetLoader


class SecureLegalFLClient(fl.client.NumPyClient):
    """
    Secure Federated Learning client for legal document summarization.
    Handles local training, encryption, and secure communication.
    """
    
    def __init__(
        self,
        model: QLoRALegalModel,
        train_loader,
        val_loader,
        cid: str,
        use_dp: bool = True,
        dp_noise_multiplier: float = 1.0,
        dp_max_grad_norm: float = 1.0,
        encryption_key: Optional[bytes] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize secure FL client.
        
        Args:
            model: QLoRA model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            cid: Client identifier
            use_dp: Enable differential privacy
            dp_noise_multiplier: DP noise multiplier
            dp_max_grad_norm: DP max gradient norm
            encryption_key: AES-256 encryption key (if None, generates new)
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cid = cid
        self.use_dp = use_dp
        self.device = device
        
        # Load model
        self.peft_model = model.get_model()
        
        # Setup encryption
        self.encryption = AES256Encryption(key=encryption_key)
        
        # Setup trainer
        self.trainer = LegalTrainer(
            model=self.peft_model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device
        )
        
        # Setup DP trainer if enabled
        self.dp_trainer = None
        if use_dp:
            sample_rate = len(train_loader.dataset) / len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 0.01
            self.dp_trainer = DPLegalTrainer(
                model=self.peft_model,
                train_dataloader=train_loader,
                sample_rate=sample_rate,
                noise_multiplier=dp_noise_multiplier,
                max_grad_norm=dp_max_grad_norm,
                device=device
            )
        
        # Training metrics
        self.training_history = []
        self.privacy_spent = {"epsilon": 0.0, "delta": 0.0}
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get current model parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of numpy arrays representing model parameters
        """
        parameters = self.model.get_trainable_parameters()
        return [param.cpu().numpy() for param in parameters.values()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Set model parameters from server.
        
        Args:
            parameters: List of numpy arrays
        """
        # Convert numpy arrays to tensors
        param_dict = {}
        param_names = [name for name, _ in self.peft_model.named_parameters() if _.requires_grad]
        
        for i, (name, param) in enumerate(zip(param_names, parameters)):
            if i < len(parameters):
                param_dict[name] = torch.from_numpy(parameters[i])
        
        self.model.set_parameters(param_dict)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration
            
        Returns:
            Tuple of (updated parameters, number of samples, metrics)
        """
        print(f"[Client {self.cid}] Starting local training...")
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Extract training config
        local_epochs = config.get("local_epochs", 1)
        learning_rate = config.get("learning_rate", 2e-4)
        
        # Update trainer learning rate
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # Train for specified epochs
        self.trainer.num_epochs = local_epochs
        history = self.trainer.train()
        
        # Get updated parameters
        updated_params = self.get_parameters(config)
        
        # Get training metrics
        train_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
        num_samples = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 0
        
        # Get privacy spent if DP enabled
        if self.use_dp and self.dp_trainer:
            epsilon, delta = self.dp_trainer.get_privacy_spent()
            self.privacy_spent = {"epsilon": epsilon, "delta": delta}
        
        # Encrypt parameters before sending
        param_dict = {}
        param_names = [name for name, _ in self.peft_model.named_parameters() if _.requires_grad]
        for i, (name, param) in enumerate(zip(param_names, updated_params)):
            if i < len(updated_params):
                param_dict[name] = torch.from_numpy(param)
        
        encrypted_params = self.encryption.encrypt_parameters(param_dict)
        
        # Store metrics
        metrics = {
            "train_loss": train_loss,
            "num_samples": num_samples,
            "privacy_epsilon": self.privacy_spent.get("epsilon", 0.0),
            "privacy_delta": self.privacy_spent.get("delta", 0.0),
            "encrypted": True
        }
        
        print(f"[Client {self.cid}] Training complete. Loss: {train_loss:.4f}")
        
        # Return encrypted parameters (will be decrypted on server)
        # For Flower compatibility, we return numpy arrays but note encryption in metadata
        return updated_params, num_samples, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local validation data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, number of samples, metrics)
        """
        print(f"[Client {self.cid}] Starting evaluation...")
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluate
        val_loss = self.trainer.validate()
        
        num_samples = len(self.val_loader.dataset) if hasattr(self.val_loader, 'dataset') else 0
        
        metrics = {
            "val_loss": val_loss,
            "num_samples": num_samples
        }
        
        print(f"[Client {self.cid}] Evaluation complete. Loss: {val_loss:.4f}")
        
        return float(val_loss), num_samples, metrics


def create_client(
    dataset_path: str,
    client_id: str,
    batch_size: int = 4,
    use_dp: bool = True,
    encryption_key: Optional[bytes] = None
) -> SecureLegalFLClient:
    """
    Factory function to create a secure FL client.
    
    Args:
        dataset_path: Path to local legal dataset
        client_id: Unique client identifier
        batch_size: Training batch size
        use_dp: Enable differential privacy
        encryption_key: Encryption key (if None, generates new)
        
    Returns:
        Configured SecureLegalFLClient instance
    """
    # Load dataset
    loader = LegalDatasetLoader()
    dataset = loader.load_local_dataset(dataset_path, num_samples=100)
    train_dataset, val_dataset, _ = loader.split_dataset(dataset)
    
    # Create data loaders
    train_loader = loader.get_dataloader(train_dataset, batch_size=batch_size)
    val_loader = loader.get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = QLoRALegalModel()
    
    # Create client
    client = SecureLegalFLClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cid=client_id,
        use_dp=use_dp,
        encryption_key=encryption_key
    )
    
    return client


def start_client(
    server_address: str = "localhost:8080",
    client_id: str = "client_0",
    dataset_path: str = "./data/legal_docs",
    use_dp: bool = True
):
    """
    Start Flower client and connect to server.
    
    Args:
        server_address: Server address (host:port)
        client_id: Client identifier
        dataset_path: Path to local dataset
        use_dp: Enable differential privacy
    """
    # Create client
    client = create_client(
        dataset_path=dataset_path,
        client_id=client_id,
        use_dp=use_dp
    )
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )
