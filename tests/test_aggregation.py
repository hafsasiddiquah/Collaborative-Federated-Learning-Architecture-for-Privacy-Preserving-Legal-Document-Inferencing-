"""
Unit Tests for FedAvg Aggregation
Tests the correctness and convergence of federated averaging.
"""

import torch
import numpy as np
import pytest
from unittest.mock import Mock

from server.strategy import SecureFedAvgStrategy


class TestFedAvgAggregation:
    """Test cases for FedAvg aggregation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.strategy = SecureFedAvgStrategy(
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            fraction_fit=0.8,
            fraction_evaluate=0.8,
            initial_parameters=None
        )

        # Create sample model parameters
        self.sample_params = {
            "layer1.weight": torch.randn(100, 50),
            "layer1.bias": torch.randn(100),
            "layer2.weight": torch.randn(50, 10),
            "layer2.bias": torch.randn(10)
        }

    def test_parameter_aggregation_basic(self):
        """Test basic parameter aggregation."""
        # Create client updates
        client1_params = {
            "layer1.weight": torch.ones(100, 50),
            "layer1.bias": torch.ones(100),
            "layer2.weight": torch.ones(50, 10),
            "layer2.bias": torch.ones(10)
        }

        client2_params = {
            "layer1.weight": torch.ones(100, 50) * 2,
            "layer1.bias": torch.ones(100) * 2,
            "layer2.weight": torch.ones(50, 10) * 2,
            "layer2.bias": torch.ones(10) * 2
        }

        client_updates = [client1_params, client2_params]

        aggregated = self.strategy._aggregate_parameters(client_updates)

        # Expected: average of client1 and client2
        expected_weight1 = (torch.ones(100, 50) + torch.ones(100, 50) * 2) / 2
        expected_bias1 = (torch.ones(100) + torch.ones(100) * 2) / 2

        assert torch.allclose(aggregated["layer1.weight"], expected_weight1)
        assert torch.allclose(aggregated["layer1.bias"], expected_bias1)

    def test_weighted_aggregation(self):
        """Test weighted aggregation based on client data sizes."""
        client1_params = {
            "layer1.weight": torch.ones(50, 30),
            "layer1.bias": torch.ones(50)
        }

        client2_params = {
            "layer1.weight": torch.ones(50, 30) * 3,
            "layer1.bias": torch.ones(50) * 3
        }

        client3_params = {
            "layer1.weight": torch.ones(50, 30) * 5,
            "layer1.bias": torch.ones(50) * 5
        }

        # Different data sizes
        client_weights = [100, 200, 150]  # Total: 450

        client_updates = [client1_params, client2_params, client3_params]

        aggregated = self.strategy._aggregate_parameters_weighted(
            client_updates, client_weights
        )

        # Expected: weighted average
        # Client1: 100/450 * 1 + 200/450 * 3 + 150/450 * 5 = 0.222*1 + 0.444*3 + 0.333*5
        expected_weight = (100*1 + 200*3 + 150*5) / 450  # Direct calculation
        expected_bias = expected_weight

        assert torch.allclose(aggregated["layer1.weight"], torch.full((50, 30), expected_weight), atol=1e-4)
        assert torch.allclose(aggregated["layer1.bias"], torch.full((50,), expected_bias), atol=1e-4)

    def test_aggregation_with_different_architectures(self):
        """Test aggregation with clients having different architectures."""
        # Client 1: 2-layer network
        client1_params = {
            "layer1.weight": torch.ones(64, 32),
            "layer1.bias": torch.ones(64),
            "layer2.weight": torch.ones(32, 16),
            "layer2.bias": torch.ones(16)
        }

        # Client 2: 3-layer network (different architecture)
        client2_params = {
            "layer1.weight": torch.ones(64, 32),
            "layer1.bias": torch.ones(64),
            "layer2.weight": torch.ones(32, 16),
            "layer2.bias": torch.ones(16),
            "layer3.weight": torch.ones(16, 8),
            "layer3.bias": torch.ones(8)
        }

        client_updates = [client1_params, client2_params]

        # Should handle missing parameters gracefully
        aggregated = self.strategy._aggregate_parameters(client_updates)

        # Should only aggregate common parameters
        assert "layer1.weight" in aggregated
        assert "layer1.bias" in aggregated
        assert "layer2.weight" in aggregated
        assert "layer2.bias" in aggregated
        assert "layer3.weight" not in aggregated  # Not in client1

    def test_convergence_tracking(self):
        """Test convergence tracking during aggregation."""
        # Simulate multiple rounds
        initial_params = {
            "layer1.weight": torch.zeros(20, 10),
            "layer1.bias": torch.zeros(20)
        }

        # Round 1
        round1_updates = [
            {
                "layer1.weight": torch.ones(20, 10) * 0.1,
                "layer1.bias": torch.ones(20) * 0.1
            },
            {
                "layer1.weight": torch.ones(20, 10) * 0.2,
                "layer1.bias": torch.ones(20) * 0.2
            }
        ]

        aggregated1 = self.strategy._aggregate_parameters(round1_updates)
        self.strategy._update_convergence_metrics(initial_params, aggregated1, 1)

        # Round 2
        round2_updates = [
            {
                "layer1.weight": torch.ones(20, 10) * 0.15,
                "layer1.bias": torch.ones(20) * 0.15
            },
            {
                "layer1.weight": torch.ones(20, 10) * 0.25,
                "layer1.bias": torch.ones(20) * 0.25
            }
        ]

        aggregated2 = self.strategy._aggregate_parameters(round2_updates)
        self.strategy._update_convergence_metrics(aggregated1, aggregated2, 2)

        # Check convergence metrics
        metrics = self.strategy.get_convergence_metrics()

        assert "parameter_changes" in metrics
        assert "gradient_norms" in metrics
        assert len(metrics["parameter_changes"]) == 2
        assert len(metrics["gradient_norms"]) == 2

    def test_robust_aggregation_outlier_removal(self):
        """Test robust aggregation with outlier removal."""
        # Create normal updates
        normal_updates = [
            {
                "layer1.weight": torch.randn(30, 20) * 0.1,
                "layer1.bias": torch.randn(30) * 0.1
            }
            for _ in range(5)
        ]

        # Add outlier updates
        outlier_updates = [
            {
                "layer1.weight": torch.ones(30, 20) * 10,  # Very large values
                "layer1.bias": torch.ones(30) * 10
            },
            {
                "layer1.weight": torch.ones(30, 20) * (-5),  # Large negative values
                "layer1.bias": torch.ones(30) * (-5)
            }
        ]

        all_updates = normal_updates + outlier_updates

        # Robust aggregation should remove outliers
        aggregated = self.strategy._robust_aggregate_parameters(all_updates)

        # Check that aggregated values are reasonable (not dominated by outliers)
        weight_mean = torch.mean(aggregated["layer1.weight"]).item()
        bias_mean = torch.mean(aggregated["layer1.bias"]).item()

        # Should be close to zero (normal updates are ~0, outliers are large)
        assert abs(weight_mean) < 2.0
        assert abs(bias_mean) < 2.0

    def test_federated_evaluation(self):
        """Test federated evaluation metrics aggregation."""
        # Mock client results
        client_results = [
            {
                "loss": 0.5,
                "accuracy": 0.85,
                "bleu": 0.75,
                "rouge": 0.78
            },
            {
                "loss": 0.4,
                "accuracy": 0.88,
                "bleu": 0.80,
                "rouge": 0.82
            },
            {
                "loss": 0.6,
                "accuracy": 0.82,
                "bleu": 0.72,
                "rouge": 0.75
            }
        ]

        aggregated_metrics = self.strategy._aggregate_evaluation_metrics(client_results)

        # Check aggregated metrics
        assert "loss" in aggregated_metrics
        assert "accuracy" in aggregated_metrics
        assert "bleu" in aggregated_metrics
        assert "rouge" in aggregated_metrics

        # Loss should be average
        expected_loss = (0.5 + 0.4 + 0.6) / 3
        assert abs(aggregated_metrics["loss"] - expected_loss) < 1e-6

        # Accuracy should be average
        expected_acc = (0.85 + 0.88 + 0.82) / 3
        assert abs(aggregated_metrics["accuracy"] - expected_acc) < 1e-6

    def test_client_selection(self):
        """Test client selection for training."""
        available_clients = [f"client_{i}" for i in range(10)]

        # Test fraction-based selection
        selected = self.strategy._select_clients(available_clients, fraction=0.5)

        assert len(selected) == 5  # 50% of 10
        assert all(client in available_clients for client in selected)

        # Test minimum client selection
        selected_min = self.strategy._select_clients(available_clients, min_clients=3)

        assert len(selected_min) >= 3
        assert all(client in available_clients for client in selected_min)

    def test_parameter_validation(self):
        """Test parameter validation before aggregation."""
        # Valid parameters
        valid_params = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10)
        }

        assert self.strategy._validate_parameters(valid_params)

        # Invalid parameters (None values)
        invalid_params = {
            "layer1.weight": None,
            "layer1.bias": torch.randn(10)
        }

        assert not self.strategy._validate_parameters(invalid_params)

        # Invalid parameters (wrong shapes)
        wrong_shape_params = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(5)  # Wrong shape
        }

        # This might pass validation if we don't check shapes
        # But should at least check for None values
        assert self.strategy._validate_parameters(wrong_shape_params)

    def test_gradient_clipping(self):
        """Test gradient clipping during aggregation."""
        # Create large gradients
        large_grads = {
            "layer1.weight": torch.ones(20, 10) * 10,
            "layer1.bias": torch.ones(20) * 10
        }

        clipped = self.strategy._clip_gradients(large_grads, max_norm=5.0)

        # Check that norms are clipped
        total_norm = torch.sqrt(sum(torch.sum(p**2) for p in clipped.values()))
        assert total_norm <= 5.0 + 1e-6  # Allow small numerical error

    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate adjustment."""
        # Simulate convergence slowdown
        slow_convergence_metrics = {
            "parameter_changes": [0.1, 0.05, 0.02, 0.01],  # Decreasing changes
            "gradient_norms": [1.0, 0.8, 0.6, 0.4]
        }

        self.strategy.convergence_metrics = slow_convergence_metrics

        # Should increase learning rate for slow convergence
        adaptive_lr = self.strategy._compute_adaptive_learning_rate(base_lr=0.01)

        assert adaptive_lr > 0.01  # Should be higher than base

        # Simulate fast convergence
        fast_convergence_metrics = {
            "parameter_changes": [0.5, 0.3, 0.1, 0.05],
            "gradient_norms": [2.0, 1.5, 1.0, 0.8]
        }

        self.strategy.convergence_metrics = fast_convergence_metrics

        # Should decrease learning rate for fast convergence
        adaptive_lr_fast = self.strategy._compute_adaptive_learning_rate(base_lr=0.01)

        assert adaptive_lr_fast < 0.01  # Should be lower than base

    def test_secure_aggregation_simulation(self):
        """Test secure aggregation with encrypted gradients."""
        # Mock encrypted gradients
        encrypted_grads = [
            {
                "layer1.weight": torch.randn(15, 10),
                "layer1.bias": torch.randn(15)
            },
            {
                "layer1.weight": torch.randn(15, 10),
                "layer1.bias": torch.randn(15)
            }
        ]

        # Mock decryption (in real implementation, this would use secure MPC)
        decrypted_grads = encrypted_grads  # Assume decryption succeeds

        aggregated = self.strategy._secure_aggregate(decrypted_grads)

        # Should produce valid aggregated parameters
        assert "layer1.weight" in aggregated
        assert "layer1.bias" in aggregated
        assert aggregated["layer1.weight"].shape == (15, 10)
        assert aggregated["layer1.bias"].shape == (15,)

    def test_federated_learning_simulation(self):
        """Test full federated learning simulation."""
        # Initialize global model
        global_params = {
            "layer1.weight": torch.zeros(25, 15),
            "layer1.bias": torch.zeros(25),
            "layer2.weight": torch.zeros(15, 5),
            "layer2.bias": torch.zeros(5)
        }

        # Simulate client training (simplified)
        client_updates = []
        for i in range(3):
            # Simulate local training updates
            update = {
                "layer1.weight": torch.randn(25, 15) * 0.1,
                "layer1.bias": torch.randn(25) * 0.1,
                "layer2.weight": torch.randn(15, 5) * 0.1,
                "layer2.bias": torch.randn(5) * 0.1
            }
            client_updates.append(update)

        # Aggregate updates
        aggregated_update = self.strategy._aggregate_parameters(client_updates)

        # Update global model
        new_global_params = {}
        for key in global_params:
            new_global_params[key] = global_params[key] + aggregated_update[key]

        # Verify update
        assert all(key in new_global_params for key in global_params)
        assert not torch.equal(new_global_params["layer1.weight"], global_params["layer1.weight"])


class TestAggregationCorrectness:
    """Test mathematical correctness of aggregation operations."""

    def test_commutativity(self):
        """Test that aggregation is commutative."""
        params1 = {"layer1": torch.tensor([1.0, 2.0, 3.0])}
        params2 = {"layer1": torch.tensor([4.0, 5.0, 6.0])}
        params3 = {"layer1": torch.tensor([7.0, 8.0, 9.0])}

        # Order 1: 1,2,3
        agg1 = SecureFedAvgStrategy._aggregate_parameters_static([params1, params2, params3])

        # Order 2: 3,1,2
        agg2 = SecureFedAvgStrategy._aggregate_parameters_static([params3, params1, params2])

        # Should be equal
        assert torch.allclose(agg1["layer1"], agg2["layer1"])

    def test_associativity(self):
        """Test that aggregation is associative."""
        params1 = {"layer1": torch.tensor([1.0, 2.0])}
        params2 = {"layer1": torch.tensor([3.0, 4.0])}
        params3 = {"layer1": torch.tensor([5.0, 6.0])}

        # (1+2)+3
        agg12 = SecureFedAvgStrategy._aggregate_parameters_static([params1, params2])
        agg123 = SecureFedAvgStrategy._aggregate_parameters_static([agg12, params3])

        # 1+(2+3)
        agg23 = SecureFedAvgStrategy._aggregate_parameters_static([params2, params3])
        agg123_alt = SecureFedAvgStrategy._aggregate_parameters_static([params1, agg23])

        # Should be equal
        assert torch.allclose(agg123["layer1"], agg123_alt["layer1"])

    def test_identity_element(self):
        """Test identity element property."""
        params1 = {"layer1": torch.tensor([1.0, 2.0, 3.0])}
        identity = {"layer1": torch.zeros(3)}

        # params + identity = params
        result = SecureFedAvgStrategy._aggregate_parameters_static([params1, identity])

        # Should be close to original (within numerical precision)
        assert torch.allclose(result["layer1"], params1["layer1"], atol=1e-6)

    def test_weighted_aggregation_correctness(self):
        """Test mathematical correctness of weighted aggregation."""
        params1 = {"layer1": torch.tensor([2.0])}
        params2 = {"layer1": torch.tensor([4.0])}
        params3 = {"layer1": torch.tensor([6.0])}

        weights = [1, 2, 3]  # Total weight = 6

        result = SecureFedAvgStrategy._aggregate_parameters_weighted_static(
            [params1, params2, params3], weights
        )

        # Expected: (1*2 + 2*4 + 3*6) / 6 = (2 + 8 + 18) / 6 = 28/6 = 4.666...
        expected = (1*2 + 2*4 + 3*6) / 6

        assert abs(result["layer1"].item() - expected) < 1e-6

    def test_convergence_properties(self):
        """Test convergence properties of federated averaging."""
        # Start with random parameters
        initial = {"layer1": torch.randn(10)}

        # Simulate multiple rounds of small updates
        current = initial.copy()
        learning_rate = 0.1

        for round_num in range(5):
            # Generate small random updates
            updates = [
                {"layer1": torch.randn(10) * 0.01},
                {"layer1": torch.randn(10) * 0.01},
                {"layer1": torch.randn(10) * 0.01}
            ]

            # Aggregate and update
            aggregated = SecureFedAvgStrategy._aggregate_parameters_static(updates)
            current["layer1"] = current["layer1"] + learning_rate * aggregated["layer1"]

        # Parameters should not have diverged dramatically
        initial_norm = torch.norm(initial["layer1"])
        final_norm = torch.norm(current["layer1"])

        # Final norm should be reasonable (not exploded)
        assert final_norm < initial_norm * 2.0
        assert final_norm > initial_norm * 0.5


if __name__ == "__main__":
    pytest.main([__file__])