"""
Local Training Module for Legal Document Summarization
Includes training loop with gradient accumulation and evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from typing import Dict, Optional, Tuple
import numpy as np


class LegalTrainer:
    """
    Trainer for fine-tuning legal document summarization models.
    Supports gradient accumulation, mixed precision, and evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 100,
        use_fp16: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps for scheduler
            use_fp16: Use mixed precision training
            device: Device to train on
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.use_fp16 = use_fp16
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Calculate total training steps
        self.total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        
        # Setup learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Setup mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_fp16 and device == "cuda" else None
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": []
        }
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch+1}/{self.num_epochs}",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_fp16 and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_fp16 and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_fp16 and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "lr": f"{current_lr:.2e}"
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(self) -> Dict[str, list]:
        """
        Run full training loop.
        
        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.device}")
        print(f"Total steps: {self.total_steps}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        for epoch in range(self.num_epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            
            # Validate
            if self.val_dataloader is not None:
                val_loss = self.validate()
                self.history["val_loss"].append(val_loss)
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
            
            # Save learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            self.history["learning_rates"].append(current_lr)
        
        return self.history
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Extract gradients from model parameters.
        Used for federated learning gradient transmission.
        
        Returns:
            Dictionary of parameter names to gradient tensors
        """
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.data.clone()
        return gradients
    
    def apply_gradients(self, gradients: Dict[str, torch.Tensor]):
        """
        Apply gradients to model parameters.
        
        Args:
            gradients: Dictionary of parameter names to gradient tensors
        """
        for name, param in self.model.named_parameters():
            if name in gradients:
                param.grad = gradients[name].clone()
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get current model parameters (for federated learning).
        
        Returns:
            Dictionary of parameter names to parameter tensors
        """
        parameters = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                parameters[name] = param.data.clone()
        return parameters
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set model parameters (for federated learning).
        
        Args:
            parameters: Dictionary of parameter names to parameter tensors
        """
        for name, param in self.model.named_parameters():
            if name in parameters:
                param.data = parameters[name].clone()
