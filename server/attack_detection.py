"""
Advanced Poisoning Attack Detection for Federated Learning
Implements multiple detection mechanisms for gradient poisoning.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from scipy.stats import zscore


class PoisoningDetector:
    """
    Multi-layered poisoning attack detection system.
    Combines statistical analysis, clustering, and norm-based filtering.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        norm_threshold: float = 10.0,
        zscore_threshold: float = 3.0,
        use_clustering: bool = True,
        cluster_method: str = "kmeans"
    ):
        """
        Initialize poisoning detector.

        Args:
            similarity_threshold: Minimum cosine similarity to accept
            norm_threshold: Maximum gradient norm threshold
            zscore_threshold: Z-score threshold for outlier detection
            use_clustering: Whether to use clustering-based detection
            cluster_method: Clustering method ("kmeans" or "dbscan")
        """
        self.similarity_threshold = similarity_threshold
        self.norm_threshold = norm_threshold
        self.zscore_threshold = zscore_threshold
        self.use_clustering = use_clustering
        self.cluster_method = cluster_method

        # History for adaptive thresholds
        self.norm_history = []
        self.similarity_history = []

    def detect_poisoning(
        self,
        gradients_list: List[Dict[str, torch.Tensor]],
        client_ids: List[str]
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[str], Dict[str, List[str]]]:
        """
        Detect and filter poisoned gradients using multiple methods.

        Args:
            gradients_list: List of gradient dictionaries from clients
            client_ids: List of client identifiers

        Returns:
            Tuple of (filtered_gradients, filtered_ids, detection_results)
        """
        if len(gradients_list) < 2:
            return gradients_list, client_ids, {"filtered": [], "reasons": []}

        # Convert gradients to vectors for analysis
        grad_vectors = self._gradients_to_vectors(gradients_list)

        # Apply multiple detection methods
        detection_results = {}

        # Method 1: Cosine similarity filtering
        similarity_scores = self._compute_similarity_scores(grad_vectors)
        similarity_filtered = self._filter_by_similarity(
            similarity_scores, self.similarity_threshold
        )

        # Method 2: Gradient norm filtering
        norm_scores = self._compute_norm_scores(grad_vectors)
        norm_filtered = self._filter_by_norm(norm_scores, self.norm_threshold)

        # Method 3: Z-score outlier detection
        zscore_filtered = self._filter_by_zscore(grad_vectors)

        # Method 4: Clustering-based detection (optional)
        cluster_filtered = []
        if self.use_clustering:
            cluster_filtered = self._filter_by_clustering(grad_vectors)

        # Combine all detection results
        all_filtered_indices = set(range(len(gradients_list)))

        # Apply similarity filter
        all_filtered_indices &= set(similarity_filtered)

        # Apply norm filter
        all_filtered_indices &= set(norm_filtered)

        # Apply z-score filter
        all_filtered_indices &= set(zscore_filtered)

        # Apply clustering filter
        if cluster_filtered:
            all_filtered_indices &= set(cluster_filtered)

        # Ensure at least one client remains
        if not all_filtered_indices:
            # Fallback: keep clients with highest average score
            scores = self._compute_combined_scores(
                similarity_scores, norm_scores, grad_vectors
            )
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            all_filtered_indices = set(sorted_indices[:max(1, len(sorted_indices) // 2)])

        filtered_indices = sorted(list(all_filtered_indices))
        filtered_gradients = [gradients_list[i] for i in filtered_indices]
        filtered_ids = [client_ids[i] for i in filtered_indices]

        # Compile detection results
        detection_results = {
            "similarity_filtered": [client_ids[i] for i in range(len(client_ids))
                                  if i not in similarity_filtered],
            "norm_filtered": [client_ids[i] for i in range(len(client_ids))
                            if i not in norm_filtered],
            "zscore_filtered": [client_ids[i] for i in range(len(client_ids))
                              if i not in zscore_filtered],
            "cluster_filtered": [client_ids[i] for i in range(len(client_ids))
                               if i not in cluster_filtered] if cluster_filtered else [],
            "final_filtered": [client_ids[i] for i in range(len(client_ids))
                             if i not in filtered_indices]
        }

        # Update history for adaptive thresholds
        self._update_history(norm_scores, similarity_scores)

        return filtered_gradients, filtered_ids, detection_results

    def _gradients_to_vectors(self, gradients_list: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        """
        Convert gradient dictionaries to vectors for analysis.

        Args:
            gradients_list: List of gradient dictionaries

        Returns:
            Array of gradient vectors
        """
        vectors = []
        for grads in gradients_list:
            # Flatten all gradients into a single vector
            flat_grads = []
            for name, grad in grads.items():
                flat_grads.append(grad.flatten().cpu().numpy())
            vector = np.concatenate(flat_grads)
            vectors.append(vector)

        return np.array(vectors)

    def _compute_similarity_scores(self, grad_vectors: np.ndarray) -> np.ndarray:
        """
        Compute average cosine similarity scores for each client.

        Args:
            grad_vectors: Gradient vectors

        Returns:
            Array of similarity scores
        """
        if len(grad_vectors) < 2:
            return np.ones(len(grad_vectors))

        similarity_matrix = cosine_similarity(grad_vectors)
        avg_similarities = []

        for i in range(len(grad_vectors)):
            similarities = [similarity_matrix[i, j] for j in range(len(grad_vectors)) if i != j]
            avg_sim = np.mean(similarities)
            avg_similarities.append(avg_sim)

        return np.array(avg_similarities)

    def _compute_norm_scores(self, grad_vectors: np.ndarray) -> np.ndarray:
        """
        Compute L2 norm scores for gradient vectors.

        Args:
            grad_vectors: Gradient vectors

        Returns:
            Array of norm scores
        """
        norms = np.linalg.norm(grad_vectors, axis=1)
        return norms

    def _filter_by_similarity(
        self,
        similarity_scores: np.ndarray,
        threshold: float
    ) -> List[int]:
        """
        Filter clients by similarity threshold.

        Args:
            similarity_scores: Similarity scores
            threshold: Minimum similarity threshold

        Returns:
            List of accepted client indices
        """
        return [i for i, score in enumerate(similarity_scores) if score >= threshold]

    def _filter_by_norm(
        self,
        norm_scores: np.ndarray,
        threshold: float
    ) -> List[int]:
        """
        Filter clients by gradient norm threshold.

        Args:
            norm_scores: Norm scores
            threshold: Maximum norm threshold

        Returns:
            List of accepted client indices
        """
        return [i for i, score in enumerate(norm_scores) if score <= threshold]

    def _filter_by_zscore(self, grad_vectors: np.ndarray) -> List[int]:
        """
        Filter clients using z-score outlier detection.

        Args:
            grad_vectors: Gradient vectors

        Returns:
            List of accepted client indices
        """
        # Compute z-scores for each dimension
        z_scores = zscore(grad_vectors, axis=0)

        # Compute maximum z-score for each client
        max_z_scores = np.max(np.abs(z_scores), axis=1)

        # Filter out outliers
        return [i for i, z in enumerate(max_z_scores) if z <= self.zscore_threshold]

    def _filter_by_clustering(self, grad_vectors: np.ndarray) -> List[int]:
        """
        Filter clients using clustering-based detection.

        Args:
            grad_vectors: Gradient vectors

        Returns:
            List of accepted client indices
        """
        if len(grad_vectors) < 3:
            return list(range(len(grad_vectors)))

        try:
            if self.cluster_method == "kmeans":
                # Use K-means clustering
                kmeans = KMeans(n_clusters=min(3, len(grad_vectors)), random_state=42)
                labels = kmeans.fit_predict(grad_vectors)

                # Find the largest cluster
                unique_labels, counts = np.unique(labels, return_counts=True)
                majority_cluster = unique_labels[np.argmax(counts)]

                # Keep only clients in the majority cluster
                return [i for i, label in enumerate(labels) if label == majority_cluster]

            else:
                # Default: keep all if clustering fails
                return list(range(len(grad_vectors)))

        except Exception as e:
            print(f"Clustering failed: {e}")
            return list(range(len(grad_vectors)))

    def _compute_combined_scores(
        self,
        similarity_scores: np.ndarray,
        norm_scores: np.ndarray,
        grad_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute combined detection scores.

        Args:
            similarity_scores: Similarity scores
            norm_scores: Norm scores
            grad_vectors: Gradient vectors

        Returns:
            Combined scores
        """
        # Normalize scores
        norm_sim = (similarity_scores - np.min(similarity_scores)) / \
                  (np.max(similarity_scores) - np.min(similarity_scores) + 1e-8)

        norm_norms = 1 - (norm_scores - np.min(norm_scores)) / \
                    (np.max(norm_scores) - np.min(norm_scores) + 1e-8)

        # Combine with weights
        combined = 0.6 * norm_sim + 0.4 * norm_norms

        return combined

    def _update_history(
        self,
        norm_scores: np.ndarray,
        similarity_scores: np.ndarray
    ):
        """
        Update detection history for adaptive thresholds.

        Args:
            norm_scores: Current norm scores
            similarity_scores: Current similarity scores
        """
        self.norm_history.extend(norm_scores)
        self.similarity_history.extend(similarity_scores)

        # Keep only recent history
        max_history = 1000
        if len(self.norm_history) > max_history:
            self.norm_history = self.norm_history[-max_history:]
        if len(self.similarity_history) > max_history:
            self.similarity_history = self.similarity_history[-max_history:]

    def get_adaptive_thresholds(self) -> Dict[str, float]:
        """
        Compute adaptive thresholds based on history.

        Returns:
            Dictionary with adaptive thresholds
        """
        thresholds = {
            "similarity_threshold": self.similarity_threshold,
            "norm_threshold": self.norm_threshold
        }

        if len(self.similarity_history) > 10:
            # Set similarity threshold to 25th percentile
            thresholds["similarity_threshold"] = np.percentile(
                self.similarity_history, 25
            )

        if len(self.norm_history) > 10:
            # Set norm threshold to 75th percentile
            thresholds["norm_threshold"] = np.percentile(
                self.norm_history, 75
            )

        return thresholds

    def reset_history(self):
        """Reset detection history."""
        self.norm_history = []
        self.similarity_history = []