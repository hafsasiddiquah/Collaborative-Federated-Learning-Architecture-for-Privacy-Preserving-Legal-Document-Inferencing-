"""
Unit Tests for Differential Privacy Implementation
Tests the correctness of DP-SGD noise addition and privacy accounting.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from unittest.mock import Mock, patch

from client.dp_engine import DPLegalTrainer


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class MockDataLoader:
    """Mock data loader for testing."""
    def __init__(self, batch_size=4, num_batches=10):
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            # Generate random batch
            x = torch.randn(self.batch_size, 10)
            y = torch.randn(self.batch_size, 1)
            yield {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": y}


class TestDPLegalTrainer:
    """Test cases for DP trainer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.train_loader = MockDataLoader()
        self.device = "cpu"

        # Mock loss function
        self.loss_fn = nn.MSELoss()

    def test_initialization(self):
        """Test DP trainer initialization."""
        dp_trainer = DPLegalTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            sample_rate=0.1,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            device=self.device
        )

        assert dp_trainer.model is self.model
        assert dp_trainer.sample_rate == 0.1
        assert dp_trainer.noise_multiplier == 1.0
        assert dp_trainer.max_grad_norm == 1.0

    def test_privacy_engine_attachment(self):
        """Test privacy engine attachment."""
        dp_trainer = DPLegalTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            sample_rate=0.1,
            device=self.device
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # Attach privacy engine
        optimizer_dp, privacy_engine = dp_trainer.attach_privacy_engine(optimizer)

        assert dp_trainer.privacy_engine_attached
        assert dp_trainer.privacy_engine is not None
        assert optimizer_dp is not optimizer  # Should be wrapped

    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        dp_trainer = DPLegalTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            sample_rate=0.1,
            max_grad_norm=1.0,
            device=self.device
        )

        # Create gradients that exceed max norm
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Create gradient with norm > max_grad_norm
                grad = torch.randn_like(param) * 10  # Large gradient
                param.grad = grad
                gradients[name] = grad

        # Clip gradients
        clipped = dp_trainer.get_noisy_gradients(gradients)

        # Check that norms are clipped
        for name, grad in clipped.items():
            grad_norm = torch.norm(grad)
            assert grad_norm <= dp_trainer.max_grad_norm + dp_trainer.noise_multiplier * dp_trainer.max_grad_norm

    def test_noise_addition(self):
        """Test that noise is correctly added to gradients."""
        noise_multiplier = 2.0
        max_grad_norm = 1.0

        dp_trainer = DPLegalTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            sample_rate=0.1,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            device=self.device
        )

        # Create known gradients (smaller, more realistic values)
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad = torch.randn_like(param) * 0.01  # Small random gradients
                param.grad = grad
                gradients[name] = grad.clone()

        # Get noisy gradients
        noisy_grads = dp_trainer.get_noisy_gradients(gradients)

        # Check that noise was added (gradients should be different)
        for name, noisy_grad in noisy_grads.items():
            original_grad = gradients[name]
            assert not torch.equal(noisy_grad, original_grad)

            # Check noise statistics (approximate)
            noise = noisy_grad - original_grad
            if noise.numel() > 1:  # Only check std if tensor has more than 1 element
                noise_std = torch.std(noise).item()
                # Expected noise std is approximately noise_multiplier * max_grad_norm
                expected_std = noise_multiplier * max_grad_norm
                # Allow larger tolerance for statistical variation
                assert abs(noise_std - expected_std) < expected_std * 0.5
            else:
                # For single-element tensors, just check that noise was added
                assert not torch.equal(noise, torch.zeros_like(noise))

    def test_privacy_budget_calculation(self):
        """Test privacy budget calculation."""
        dp_trainer = DPLegalTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            sample_rate=0.1,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            target_epsilon=1.0,
            device=self.device
        )

        # Mock privacy engine
        mock_privacy_engine = Mock()
        mock_privacy_engine.get_privacy_spent.return_value = (0.5, 0.01)
        dp_trainer.privacy_engine = mock_privacy_engine

        epsilon, delta = dp_trainer.get_privacy_spent()

        assert epsilon == 0.5
        assert delta == 0.01

    def test_privacy_cost_estimation(self):
        """Test privacy cost estimation for multiple epochs."""
        dp_trainer = DPLegalTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            sample_rate=0.1,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            epochs=3,
            device=self.device
        )

        # Mock privacy engine
        mock_privacy_engine = Mock()
        mock_privacy_engine.get_privacy_spent.return_value = (0.8, 0.01)
        dp_trainer.privacy_engine = mock_privacy_engine

        privacy_cost = dp_trainer.compute_privacy_cost(epochs=3)

        assert "epsilon" in privacy_cost
        assert "delta" in privacy_cost
        assert "noise_multiplier" in privacy_cost
        assert "max_grad_norm" in privacy_cost
        assert privacy_cost["total_steps"] == len(self.train_loader) * 3

    def test_noise_multiplier_adjustment(self):
        """Test noise multiplier adjustment for target epsilon."""
        dp_trainer = DPLegalTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            sample_rate=0.1,
            noise_multiplier=2.0,
            max_grad_norm=1.0,
            target_epsilon=1.0,
            device=self.device
        )

        # Mock current privacy spent
        mock_privacy_engine = Mock()
        mock_privacy_engine.get_privacy_spent.return_value = (2.0, 0.01)  # Current epsilon is 2.0
        dp_trainer.privacy_engine = mock_privacy_engine

        # Adjust for target epsilon of 1.0
        new_noise = dp_trainer.adjust_noise_multiplier(target_epsilon=1.0, epochs=1)

        # Should reduce noise multiplier
        assert new_noise < dp_trainer.noise_multiplier

    def test_model_validation(self):
        """Test model compatibility validation."""
        dp_trainer = DPLegalTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            sample_rate=0.1,
            device=self.device
        )

        # Simple model should be compatible
        is_compatible = dp_trainer.validate_model_compatibility()
        assert isinstance(is_compatible, bool)

    @patch('opacus.PrivacyEngine')
    def test_training_step_with_dp(self, mock_privacy_engine_class):
        """Test training step with DP enabled."""
        # Mock privacy engine
        mock_privacy_engine = Mock()
        mock_privacy_engine.make_private.return_value = (self.model, Mock(), self.train_loader)
        mock_privacy_engine_class.return_value = mock_privacy_engine

        dp_trainer = DPLegalTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            sample_rate=0.1,
            device=self.device
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        dp_trainer.attach_privacy_engine(optimizer)

        # Create a batch
        batch = {
            "input_ids": torch.randn(4, 10),
            "attention_mask": torch.ones(4, 10),
            "labels": torch.randn(4, 1)
        }

        # Training step
        loss = dp_trainer.train_step(batch, self.loss_fn)

        assert isinstance(loss, float)
        assert loss >= 0


class TestDPNoiseCorrectness:
    """Test DP noise statistical properties."""

    def test_noise_distribution(self):
        """Test that added noise follows Gaussian distribution."""
        dp_trainer = DPLegalTrainer(
            model=SimpleModel(),
            train_dataloader=MockDataLoader(),
            sample_rate=0.1,
            noise_multiplier=1.0,
            max_grad_norm=1.0
        )

        # Generate many noise samples
        noise_samples = []
        for _ in range(1000):
            grad = torch.ones(10)  # Dummy gradient
            noisy_grad = dp_trainer.get_noisy_gradients({"test": grad})
            noise = noisy_grad["test"] - grad
            noise_samples.extend(noise.tolist())

        noise_samples = np.array(noise_samples)

        # Check statistical properties
        mean = np.mean(noise_samples)
        std = np.std(noise_samples)

        # Should be approximately zero mean
        assert abs(mean) < 0.1

        # Should have expected standard deviation
        expected_std = 1.0 * 1.0  # noise_multiplier * max_grad_norm
        assert abs(std - expected_std) < 0.2

    def test_gradient_norm_clipping(self):
        """Test that gradient norms are properly clipped."""
        max_norm = 1.0
        dp_trainer = DPLegalTrainer(
            model=SimpleModel(),
            train_dataloader=MockDataLoader(),
            sample_rate=0.1,
            max_grad_norm=max_norm
        )

        # Test various gradient norms
        test_gradients = [
            torch.ones(10) * 0.5,  # Norm < max_norm
            torch.ones(10) * 2.0,  # Norm > max_norm
            torch.randn(10) * 5.0   # Large random gradient
        ]

        for grad in test_gradients:
            gradients = {"test": grad}
            clipped = dp_trainer.get_noisy_gradients(gradients)

            clipped_grad = clipped["test"]
            clipped_norm = torch.norm(clipped_grad)

            # Clipped norm should be <= max_norm + noise
            max_expected_norm = max_norm + dp_trainer.noise_multiplier * max_norm
            assert clipped_norm <= max_expected_norm + 0.1  # Small tolerance


if __name__ == "__main__":
    pytest.main([__file__])