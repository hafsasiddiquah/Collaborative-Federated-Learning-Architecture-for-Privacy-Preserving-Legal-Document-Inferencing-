"""
Differential Privacy Engine using Opacus
Applies Gaussian Differential Privacy to model training.
"""

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from typing import Optional, Dict, Tuple
import warnings


class DPLegalTrainer:
    """
    Differential Privacy trainer for legal document summarization.
    Uses Opacus to add Gaussian noise to gradients for privacy preservation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        sample_rate: float,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        target_epsilon: Optional[float] = None,
        target_delta: float = 1e-5,
        epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize DP trainer.
        
        Args:
            model: Model to train with DP
            train_dataloader: Training data loader
            sample_rate: Sampling rate (batch_size / dataset_size)
            noise_multiplier: Noise multiplier for Gaussian mechanism
            max_grad_norm: Maximum gradient norm for clipping
            target_epsilon: Target privacy budget (epsilon)
            target_delta: Target delta (typically 1/dataset_size)
            epochs: Number of training epochs
            device: Device to train on
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.sample_rate = sample_rate
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.epochs = epochs
        self.device = device
        
        # Validate model compatibility with Opacus
        try:
            ModuleValidator.validate(self.model, strict=False)
            self.model = ModuleValidator.fix(self.model)
        except Exception as e:
            warnings.warn(f"Model validation warning: {e}")
        
        # Initialize privacy engine
        self.privacy_engine = None
        self.optimizer = None
        self.privacy_engine_attached = False
        
        # Privacy accounting
        self.epsilon = 0.0
        self.alpha = 0.0
    
    def attach_privacy_engine(
        self,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.optim.Optimizer, PrivacyEngine]:
        """
        Attach Opacus PrivacyEngine to optimizer.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            Tuple of (optimizer with DP, privacy engine)
        """
        self.optimizer = optimizer
        
        # Create privacy engine
        self.privacy_engine = PrivacyEngine()
        
        # Attach privacy engine to model and optimizer
        self.model, self.optimizer, self.train_dataloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            batch_first=True
        )
        
        self.privacy_engine_attached = True
        print(f"Privacy engine attached. Noise multiplier: {self.noise_multiplier}")
        
        return self.optimizer, self.privacy_engine
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy budget spent (epsilon, delta).
        
        Returns:
            Tuple of (epsilon, delta)
        """
        if self.privacy_engine is None:
            return 0.0, self.target_delta
        
        epsilon, alpha = self.privacy_engine.get_privacy_spent(
            target_delta=self.target_delta
        )
        
        self.epsilon = epsilon
        self.alpha = alpha
        
        return epsilon, self.target_delta
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Optional[nn.Module] = None
    ) -> float:
        """
        Perform one training step with DP.
        
        Args:
            batch: Training batch
            loss_fn: Loss function (optional, uses model's loss if not provided)
            
        Returns:
            Loss value
        """
        if not self.privacy_engine_attached:
            raise ValueError("Privacy engine not attached. Call attach_privacy_engine() first.")
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Get loss
        if loss_fn is not None:
            loss = loss_fn(outputs, batch.get("labels"))
        else:
            loss = outputs.loss
        
        # Backward pass (Opacus handles DP automatically)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def compute_privacy_cost(
        self,
        epochs: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute privacy cost for given number of epochs.
        
        Args:
            epochs: Number of epochs (default: self.epochs)
            
        Returns:
            Dictionary with privacy metrics
        """
        if epochs is None:
            epochs = self.epochs
        
        # Approximate privacy cost using RDP accounting
        # This is a simplified calculation
        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * epochs
        
        # Get current privacy spent
        epsilon, delta = self.get_privacy_spent()
        
        # Estimate final epsilon (simplified)
        # In practice, Opacus tracks this automatically
        estimated_epsilon = epsilon * (total_steps / (steps_per_epoch * 1))
        
        return {
            "epsilon": estimated_epsilon,
            "delta": delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "sample_rate": self.sample_rate,
            "total_steps": total_steps
        }
    
    def adjust_noise_multiplier(
        self,
        target_epsilon: float,
        epochs: int
    ) -> float:
        """
        Adjust noise multiplier to achieve target epsilon.
        
        Args:
            target_epsilon: Target privacy budget
            epochs: Number of training epochs
            
        Returns:
            Recommended noise multiplier
        """
        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * epochs
        
        # Simplified calculation (in practice, use Opacus's accounting)
        # Higher noise multiplier = lower epsilon
        # This is a rough approximation
        if self.noise_multiplier > 0:
            current_epsilon = self.compute_privacy_cost(epochs)["epsilon"]
            if current_epsilon > 0:
                scale_factor = target_epsilon / current_epsilon
                recommended_noise = self.noise_multiplier * scale_factor
            else:
                recommended_noise = 1.0
        else:
            recommended_noise = 1.0
        
        return recommended_noise
    
    def get_noisy_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Gaussian noise to gradients manually (if needed).
        Note: Opacus handles this automatically during training.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Noisy gradients
        """
        noisy_gradients = {}
        
        for name, grad in gradients.items():
            # Clip gradient
            grad_norm = torch.norm(grad)
            if grad_norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / grad_norm)
            
            # Add Gaussian noise
            noise = torch.normal(
                mean=0.0,
                std=self.noise_multiplier * self.max_grad_norm,
                size=grad.shape,
                device=grad.device
            )
            
            noisy_gradients[name] = grad + noise
        
        return noisy_gradients
    
    def validate_model_compatibility(self) -> bool:
        """
        Validate that model is compatible with Opacus.
        
        Returns:
            True if compatible, False otherwise
        """
        try:
            ModuleValidator.validate(self.model, strict=True)
            return True
        except Exception as e:
            print(f"Model compatibility issue: {e}")
            return False
