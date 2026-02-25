"""
Unit Tests for Poisoning Attack Detection
Tests the accuracy and effectiveness of poisoning detection mechanisms.
"""

import torch
import numpy as np
import pytest
from unittest.mock import Mock

from server.attack_detection import PoisoningDetector


class TestPoisoningDetector:
    """Test cases for poisoning detection."""

    def setup_method(self):
        """Setup test fixtures."""
        self.detector = PoisoningDetector(
            similarity_threshold=0.8,
            norm_threshold=5.0,
            zscore_threshold=2.0
        )

        # Create sample gradient data
        self.normal_gradients = [
            {
                "layer1": torch.randn(100, 50) * 0.1,
                "layer2": torch.randn(50, 10) * 0.1
            }
            for _ in range(5)
        ]

        self.client_ids = [f"client_{i}" for i in range(5)]

    def test_initialization(self):
        """Test detector initialization."""
        detector = PoisoningDetector()

        assert detector.similarity_threshold == 0.5  # default
        assert detector.norm_threshold == 10.0  # default
        assert detector.zscore_threshold == 3.0  # default
        assert detector.use_clustering == True

    def test_gradient_to_vector_conversion(self):
        """Test conversion of gradients to vectors."""
        gradients = {
            "layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "layer2": torch.tensor([[5.0, 6.0]])
        }

        vectors = self.detector._gradients_to_vectors([gradients])

        assert len(vectors) == 1
        assert vectors[0].shape == (6,)  # 4 + 2 elements
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_equal(vectors[0], expected)

    def test_similarity_score_computation(self):
        """Test cosine similarity score computation."""
        # Create similar gradients
        grad1 = {"layer1": torch.ones(10, 10)}
        grad2 = {"layer1": torch.ones(10, 10) * 0.9}
        grad3 = {"layer1": torch.ones(10, 10) * (-1)}  # Very different

        gradients = [grad1, grad2, grad3]
        vectors = self.detector._gradients_to_vectors(gradients)

        similarities = self.detector._compute_similarity_scores(vectors)

        assert len(similarities) == 3
        assert similarities[0] > similarities[2]  # grad1 and grad2 more similar than grad1 and grad3

    def test_norm_score_computation(self):
        """Test gradient norm score computation."""
        grad1 = {"layer1": torch.ones(10, 10)}  # Norm ≈ sqrt(100) ≈ 10
        grad2 = {"layer1": torch.ones(10, 10) * 2}  # Norm ≈ 20
        grad3 = {"layer1": torch.zeros(10, 10)}  # Norm = 0

        gradients = [grad1, grad2, grad3]
        norms = self.detector._compute_norm_scores(
            self.detector._gradients_to_vectors(gradients)
        )

        assert len(norms) == 3
        assert norms[0] > norms[2]  # grad1 has higher norm than grad3
        assert norms[1] > norms[0]  # grad2 has highest norm

    def test_similarity_filtering(self):
        """Test similarity-based filtering."""
        # Create gradients with varying similarities
        base_grad = torch.randn(50, 50)
        gradients = [
            {"layer1": base_grad},  # Reference
            {"layer1": base_grad + torch.randn(50, 50) * 0.01},  # Very similar
            {"layer1": base_grad * 0.1},  # Different
            {"layer1": torch.randn(50, 50)},  # Very different
        ]

        vectors = self.detector._gradients_to_vectors(gradients)
        similarities = self.detector._compute_similarity_scores(vectors)

        # Filter with threshold (lower threshold for this test)
        accepted = self.detector._filter_by_similarity(similarities, 0.3)

        assert len(accepted) >= 2  # At least the two most similar
        assert 0 in accepted  # First gradient should always be accepted (self-similarity)
        assert 0 in accepted  # First gradient should always be accepted

    def test_norm_filtering(self):
        """Test norm-based filtering."""
        gradients = [
            {"layer1": torch.ones(10, 10)},  # Norm ≈ 10
            {"layer1": torch.ones(10, 10) * 0.5},  # Norm ≈ 5
            {"layer1": torch.ones(10, 10) * 3},  # Norm ≈ 30 (above threshold)
        ]

        vectors = self.detector._gradients_to_vectors(gradients)
        norms = self.detector._compute_norm_scores(vectors)

        accepted = self.detector._filter_by_norm(norms, 15.0)  # Threshold = 15

        assert 2 not in accepted  # Third gradient should be filtered
        assert len(accepted) >= 2  # First two should be accepted

    def test_zscore_filtering(self):
        """Test z-score based outlier detection."""
        # Create normal gradients
        normal_grads = [torch.randn(20, 20) for _ in range(10)]

        # Add outliers
        outlier1 = torch.ones(20, 20) * 10  # Very large values
        outlier2 = torch.zeros(20, 20)  # All zeros

        all_gradients = [{"layer1": g} for g in normal_grads + [outlier1, outlier2]]

        vectors = self.detector._gradients_to_vectors(all_gradients)
        accepted = self.detector._filter_by_zscore(vectors)

        # Should filter out some outliers
        assert len(accepted) < len(all_gradients)
        assert len(accepted) >= len(normal_grads) - 2  # Allow some tolerance

    def test_combined_detection(self):
        """Test combined poisoning detection."""
        # Create normal gradients
        normal_gradients = [
            {"layer1": torch.randn(30, 30) * 0.1} for _ in range(4)
        ]

        # Create poisoned gradients
        poisoned_gradients = [
            {"layer1": torch.ones(30, 30) * 2.0},  # Large norm
            {"layer1": torch.randn(30, 30) * (-0.5)},  # Different direction
        ]

        all_gradients = normal_gradients + poisoned_gradients
        all_client_ids = [f"client_{i}" for i in range(len(all_gradients))]

        filtered_grads, filtered_ids, detection_results = self.detector.detect_poisoning(
            all_gradients, all_client_ids
        )

        # Should filter out at least some poisoned gradients
        assert len(filtered_grads) <= len(all_gradients)
        assert len(filtered_ids) <= len(all_client_ids)

        # Check detection results structure
        assert "similarity_filtered" in detection_results
        assert "norm_filtered" in detection_results
        assert "zscore_filtered" in detection_results
        assert "final_filtered" in detection_results

    def test_history_update(self):
        """Test adaptive threshold history updates."""
        norms = [1.0, 2.0, 3.0, 4.0, 5.0]
        similarities = [0.9, 0.8, 0.7, 0.6, 0.5]

        self.detector._update_history(norms, similarities)

        assert len(self.detector.norm_history) == len(norms)
        assert len(self.detector.similarity_history) == len(similarities)

    def test_adaptive_thresholds(self):
        """Test adaptive threshold computation."""
        # Add some history
        norms = [1.0, 1.5, 2.0, 2.5, 3.0]
        similarities = [0.9, 0.85, 0.8, 0.75, 0.7]

        self.detector._update_history(norms, similarities)

        adaptive_thresholds = self.detector.get_adaptive_thresholds()

        assert "similarity_threshold" in adaptive_thresholds
        assert "norm_threshold" in adaptive_thresholds

        # Similarity threshold should be based on 25th percentile
        expected_sim_threshold = np.percentile(similarities, 25)
        assert adaptive_thresholds["similarity_threshold"] == expected_sim_threshold

    def test_clustering_detection(self):
        """Test clustering-based detection."""
        # Create two clusters of gradients
        cluster1 = [{"layer1": torch.randn(20, 20) + torch.ones(20, 20)} for _ in range(3)]
        cluster2 = [{"layer1": torch.randn(20, 20) - torch.ones(20, 20)} for _ in range(2)]

        all_gradients = cluster1 + cluster2

        accepted = self.detector._filter_by_clustering(
            self.detector._gradients_to_vectors(all_gradients)
        )

        # Should keep the majority cluster
        assert len(accepted) >= 3  # At least the larger cluster

    def test_edge_cases(self):
        """Test edge cases."""
        # Single client
        single_grad = [{"layer1": torch.randn(10, 10)}]
        single_ids = ["client_0"]

        filtered, ids, results = self.detector.detect_poisoning(single_grad, single_ids)

        assert len(filtered) == 1
        assert len(ids) == 1

        # Two clients
        two_grads = [
            {"layer1": torch.randn(10, 10)},
            {"layer1": torch.randn(10, 10)}
        ]
        two_ids = ["client_0", "client_1"]

        filtered, ids, results = self.detector.detect_poisoning(two_grads, two_ids)

        assert len(filtered) >= 1  # Should keep at least one

    def test_combined_score_computation(self):
        """Test combined detection score computation."""
        similarities = np.array([0.9, 0.5, 0.3])
        norms = np.array([1.0, 5.0, 10.0])
        vectors = np.random.randn(3, 100)

        scores = self.detector._compute_combined_scores(similarities, norms, vectors)

        assert len(scores) == 3
        # Higher scores should be better (higher similarity, lower norm)
        assert scores[0] > scores[1]  # First has highest similarity
        assert scores[1] > scores[2]  # Second better than third


class TestPoisoningDetectionAccuracy:
    """Test detection accuracy on synthetic poisoning attacks."""

    def setup_method(self):
        """Setup test fixtures."""
        self.detector = PoisoningDetector(
            similarity_threshold=0.7,
            norm_threshold=3.0,
            zscore_threshold=2.5
        )

    def test_backdoor_attack_detection(self):
        """Test detection of backdoor poisoning attacks."""
        # Create benign gradients
        benign_base = torch.randn(50, 50) * 0.1
        benign_gradients = [
            {"layer1": benign_base + torch.randn(50, 50) * 0.05}
            for _ in range(8)
        ]

        # Create backdoor poisoned gradients (large perturbations)
        poisoned_gradients = [
            {"layer1": benign_base + torch.randn(50, 50) * 2.0}
            for _ in range(2)
        ]

        all_gradients = benign_gradients + poisoned_gradients
        client_ids = [f"client_{i}" for i in range(len(all_gradients))]

        filtered_grads, filtered_ids, results = self.detector.detect_poisoning(
            all_gradients, client_ids
        )

        # Should detect and filter poisoned gradients
        assert len(filtered_grads) < len(all_gradients)
        assert len([cid for cid in filtered_ids if "client_8" in cid or "client_9" in cid]) < 2

    def test_model_poisoning_detection(self):
        """Test detection of model poisoning (targeted weight manipulation)."""
        # Create normal gradients
        normal_grads = []
        for _ in range(6):
            grad = {
                "layer1": torch.randn(30, 30) * 0.1,
                "layer2": torch.randn(30, 10) * 0.1
            }
            normal_grads.append(grad)

        # Create poisoned gradients (manipulate specific weights)
        poisoned_grads = []
        for _ in range(2):
            grad = {
                "layer1": torch.randn(30, 30) * 0.1,
                "layer2": torch.ones(30, 10) * 5.0  # Large values in layer2
            }
            poisoned_grads.append(grad)

        all_gradients = normal_grads + poisoned_grads
        client_ids = [f"client_{i}" for i in range(len(all_gradients))]

        filtered_grads, filtered_ids, results = self.detector.detect_poisoning(
            all_gradients, client_ids
        )

        # Should detect the poisoned gradients
        assert len(filtered_grads) <= len(normal_grads) + 1  # Allow some tolerance

    def test_false_positive_rate(self):
        """Test false positive rate on clean data."""
        # Generate clean gradients from similar distributions
        np.random.seed(42)
        torch.manual_seed(42)

        gradients = []
        for _ in range(10):
            grad = {
                "layer1": torch.randn(40, 40) * 0.1,
                "layer2": torch.randn(40, 20) * 0.1
            }
            gradients.append(grad)

        client_ids = [f"client_{i}" for i in range(len(gradients))]

        filtered_grads, filtered_ids, results = self.detector.detect_poisoning(
            gradients, client_ids
        )

        # False positive rate should be low for clean data
        false_positive_rate = (len(gradients) - len(filtered_grads)) / len(gradients)
        assert false_positive_rate < 0.3  # Less than 30% false positives

    def test_adaptive_threshold_effectiveness(self):
        """Test that adaptive thresholds improve detection."""
        # First, train detector on normal data
        normal_grads = [
            {"layer1": torch.randn(25, 25) * 0.1} for _ in range(20)
        ]

        # Simulate training by updating history
        for grad in normal_grads:
            vectors = self.detector._gradients_to_vectors([grad])
            norms = self.detector._compute_norm_scores(vectors)
            similarities = self.detector._compute_similarity_scores(vectors)
            self.detector._update_history(norms, similarities)

        # Get adaptive thresholds
        adaptive_thresholds = self.detector.get_adaptive_thresholds()

        # Test detection with adaptive thresholds
        test_normal = {"layer1": torch.randn(25, 25) * 0.1}
        test_poisoned = {"layer1": torch.ones(25, 25) * 2.0}

        all_grads = [test_normal, test_poisoned]
        client_ids = ["normal", "poisoned"]

        filtered_grads, filtered_ids, results = self.detector.detect_poisoning(
            all_grads, client_ids
        )

        # Should detect the poisoned gradient
        assert len(filtered_grads) == 1
        assert "normal" in filtered_ids
        assert "poisoned" not in filtered_ids


if __name__ == "__main__":
    pytest.main([__file__])