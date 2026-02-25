"""
Federated Learning Strategy with Poisoning Attack Detection
Implements FedAvg with cosine similarity filtering for security.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from sklearn.metrics.pairwise import cosine_similarity
import warnings


class SecureFedAvgStrategy(FedAvg):
    """
    Secure FedAvg strategy with poisoning attack detection.
    Uses cosine similarity to filter out malicious gradients.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.5,
        min_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs
    ):
        """
        Initialize secure FedAvg strategy.
        
        Args:
            similarity_threshold: Minimum cosine similarity to accept gradient
            min_clients: Minimum number of clients required
            fraction_fit: Fraction of clients used for training
            fraction_evaluate: Fraction of clients used for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients
            **kwargs: Additional arguments for FedAvg
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs
        )
        
        self.similarity_threshold = similarity_threshold
        self.min_clients = min_clients
        self.gradient_history = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException]
    ) -> Optional[Parameters]:
        """
        Aggregate model updates with poisoning detection.
        
        Args:
            server_round: Current round number
            results: List of (client, FitRes) tuples
            failures: List of failures
            
        Returns:
            Aggregated parameters or None
        """
        if not results:
            return None
        
        print(f"[Round {server_round}] Aggregating {len(results)} client updates...")
        
        # Extract parameters and metrics
        parameters_list = []
        num_examples_list = []
        client_ids = []
        
        for client, fit_res in results:
            parameters = fit_res.parameters
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics
            
            # Convert to numpy arrays
            param_arrays = parameters_to_ndarrays(parameters)
            parameters_list.append(param_arrays)
            num_examples_list.append(num_examples)
            client_ids.append(client.cid if hasattr(client, 'cid') else str(client))
        
        # Detect and filter poisoning attacks
        filtered_params, filtered_num_examples, filtered_ids = self._detect_poisoning(
            parameters_list,
            num_examples_list,
            client_ids,
            server_round
        )
        
        if not filtered_params:
            warnings.warn("All clients filtered out due to poisoning detection!")
            return None
        
        # Perform FedAvg aggregation
        aggregated_ndarrays = self._fedavg_aggregate(
            filtered_params,
            filtered_num_examples
        )
        
        # Convert back to Parameters
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        
        print(f"[Round {server_round}] Aggregation complete. "
              f"Used {len(filtered_params)}/{len(parameters_list)} clients.")
        
        return aggregated_parameters
    
    def _detect_poisoning(
        self,
        parameters_list: List[List[np.ndarray]],
        num_examples_list: List[int],
        client_ids: List[str],
        server_round: int
    ) -> Tuple[List[List[np.ndarray]], List[int], List[str]]:
        """
        Detect and filter out poisoning attacks using cosine similarity.
        
        Args:
            parameters_list: List of parameter arrays from clients
            num_examples_list: List of sample counts
            client_ids: List of client identifiers
            server_round: Current round number
            
        Returns:
            Filtered parameters, sample counts, and client IDs
        """
        if len(parameters_list) < 2:
            # Need at least 2 clients for similarity comparison
            return parameters_list, num_examples_list, client_ids
        
        # Flatten parameters to vectors for similarity computation
        param_vectors = []
        for params in parameters_list:
            # Concatenate all parameters into a single vector
            flat_params = np.concatenate([p.flatten() for p in params])
            param_vectors.append(flat_params)
        
        param_vectors = np.array(param_vectors)
        
        # Compute pairwise cosine similarities
        similarity_matrix = cosine_similarity(param_vectors)
        
        # Compute average similarity for each client
        avg_similarities = []
        for i in range(len(param_vectors)):
            # Average similarity with all other clients
            similarities = [similarity_matrix[i, j] for j in range(len(param_vectors)) if i != j]
            avg_sim = np.mean(similarities) if similarities else 0.0
            avg_similarities.append(avg_sim)
        
        # Filter clients below threshold
        filtered_indices = [
            i for i, sim in enumerate(avg_similarities)
            if sim >= self.similarity_threshold
        ]
        
        if not filtered_indices:
            # If all filtered out, keep top 50% by similarity
            sorted_indices = sorted(
                range(len(avg_similarities)),
                key=lambda i: avg_similarities[i],
                reverse=True
            )
            filtered_indices = sorted_indices[:max(1, len(sorted_indices) // 2)]
            warnings.warn(f"All clients below threshold. Keeping top {len(filtered_indices)} by similarity.")
        
        # Log filtering results
        filtered_out = [client_ids[i] for i in range(len(client_ids)) if i not in filtered_indices]
        if filtered_out:
            print(f"[Round {server_round}] Filtered out clients: {filtered_out}")
            print(f"[Round {server_round}] Similarities: {[f'{avg_similarities[i]:.3f}' for i in filtered_indices]}")
        
        # Return filtered results
        filtered_params = [parameters_list[i] for i in filtered_indices]
        filtered_num_examples = [num_examples_list[i] for i in filtered_indices]
        filtered_ids = [client_ids[i] for i in filtered_indices]
        
        return filtered_params, filtered_num_examples, filtered_ids
    
    def _fedavg_aggregate(
        self,
        parameters_list: List[List[np.ndarray]],
        num_examples_list: List[int]
    ) -> List[np.ndarray]:
        """
        Perform FedAvg aggregation weighted by number of examples.
        
        Args:
            parameters_list: List of parameter arrays
            num_examples_list: List of sample counts
            
        Returns:
            Aggregated parameters
        """
        # Normalize weights
        total_examples = sum(num_examples_list)
        weights = [n / total_examples for n in num_examples_list]
        
        # Aggregate each parameter
        aggregated = []
        for param_idx in range(len(parameters_list[0])):
            weighted_sum = np.zeros_like(parameters_list[0][param_idx])
            
            for client_idx, params in enumerate(parameters_list):
                weighted_sum += weights[client_idx] * params[param_idx]
            
            aggregated.append(weighted_sum)
        
        return aggregated
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.ClientManager
    ) -> List[Tuple[ClientProxy, Dict]]:
        """
        Configure clients for training round.
        
        Args:
            server_round: Current round number
            parameters: Current global parameters
            client_manager: Client manager
            
        Returns:
            List of (client, config) tuples
        """
        config = {
            "server_round": server_round,
            "local_epochs": 1,
            "learning_rate": 2e-4
        }
        
        return super().configure_fit(server_round, parameters, client_manager)
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.ClientManager
    ) -> List[Tuple[ClientProxy, Dict]]:
        """
        Configure clients for evaluation round.
        
        Args:
            server_round: Current round number
            parameters: Current global parameters
            client_manager: Client manager
            
        Returns:
            List of (client, config) tuples
        """
        config = {
            "server_round": server_round
        }
        
        return super().configure_evaluate(server_round, parameters, client_manager)
    
    def _aggregate_parameters(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate parameters from multiple clients (test utility method).
        
        Args:
            client_updates: List of parameter dictionaries from clients
            
        Returns:
            Aggregated parameters
        """
        if not client_updates:
            return {}
        
        # Simple average aggregation
        aggregated = {}
        for key in client_updates[0].keys():
            params = [update[key] for update in client_updates if key in update]
            if params:
                aggregated[key] = torch.stack(params).mean(dim=0)
        
        return aggregated
    
    def _aggregate_parameters_weighted(
        self, 
        client_updates: List[Dict[str, torch.Tensor]], 
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Weighted aggregation of parameters (test utility method).
        
        Args:
            client_updates: List of parameter dictionaries
            client_weights: Weights for each client
            
        Returns:
            Weighted aggregated parameters
        """
        if not client_updates or not client_weights or len(client_updates) != len(client_weights):
            return {}
        
        total_weight = sum(client_weights)
        if total_weight == 0:
            return self._aggregate_parameters(client_updates)
        
        aggregated = {}
        for key in client_updates[0].keys():
            weighted_sum = torch.zeros_like(client_updates[0][key])
            for update, weight in zip(client_updates, client_weights):
                if key in update:
                    weighted_sum += update[key] * weight
            
            aggregated[key] = weighted_sum / total_weight
        
        return aggregated
    
    def _robust_aggregate_parameters(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Robust aggregation with outlier removal (test utility method).
        
        Args:
            client_updates: List of parameter dictionaries
            
        Returns:
            Robustly aggregated parameters
        """
        # Simple implementation: remove outliers based on norm
        if len(client_updates) <= 2:
            return self._aggregate_parameters(client_updates)
        
        # Calculate norms for each update
        norms = []
        for update in client_updates:
            total_norm = sum(torch.norm(param) ** 2 for param in update.values()) ** 0.5
            norms.append(total_norm.item())
        
        # Remove outliers (simple: remove highest and lowest)
        sorted_indices = sorted(range(len(norms)), key=lambda i: norms[i])
        keep_indices = sorted_indices[1:-1]  # Remove highest and lowest
        
        filtered_updates = [client_updates[i] for i in keep_indices]
        return self._aggregate_parameters(filtered_updates)
    
    def _aggregate_evaluation_metrics(self, client_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate evaluation metrics from clients (test utility method).
        
        Args:
            client_results: List of metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        if not client_results:
            return {}
        
        aggregated = {}
        keys = client_results[0].keys()
        
        for key in keys:
            values = [result.get(key, 0.0) for result in client_results if key in result]
            if values:
                aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    def _select_clients(self, available_clients: List[str], fraction: float = 1.0, min_clients: int = 1) -> List[str]:
        """
        Select subset of clients for training (test utility method).
        
        Args:
            available_clients: List of available client IDs
            fraction: Fraction to select
            min_clients: Minimum number to select
            
        Returns:
            Selected client IDs
        """
        num_to_select = max(min_clients, int(len(available_clients) * fraction))
        num_to_select = min(num_to_select, len(available_clients))
        
        # Simple random selection
        import random
        return random.sample(available_clients, num_to_select)
    
    def _validate_parameters(self, parameters: Dict[str, torch.Tensor]) -> bool:
        """
        Validate parameter dictionary (test utility method).
        
        Args:
            parameters: Parameter dictionary to validate
            
        Returns:
            True if valid
        """
        if not parameters:
            return False
        
        for key, param in parameters.items():
            if param is None or torch.isnan(param).any() or torch.isinf(param).any():
                return False
        
        return True
    
    def _clip_gradients(self, gradients: Dict[str, torch.Tensor], max_norm: float) -> Dict[str, torch.Tensor]:
        """
        Clip gradients by global norm (test utility method).
        
        Args:
            gradients: Gradient dictionary
            max_norm: Maximum norm
            
        Returns:
            Clipped gradients
        """
        # Calculate total norm
        total_norm = sum(torch.norm(grad) ** 2 for grad in gradients.values()) ** 0.5
        
        if total_norm > max_norm:
            scale = max_norm / total_norm
            return {key: grad * scale for key, grad in gradients.items()}
        
        return gradients
    
    def _compute_adaptive_learning_rate(self, base_lr: float) -> float:
        """
        Compute adaptive learning rate based on convergence (test utility method).
        
        Args:
            base_lr: Base learning rate
            
        Returns:
            Adaptive learning rate
        """
        if not hasattr(self, 'convergence_metrics') or not self.convergence_metrics.get('parameter_changes'):
            return base_lr
        
        changes = self.convergence_metrics['parameter_changes']
        recent_change = changes[-1] if changes else 0.0
        
        # Increase LR if convergence is slow, decrease if fast
        if recent_change > 0.01:  # Slow convergence
            return base_lr * 1.2
        elif recent_change < 0.001:  # Fast convergence
            return base_lr * 0.8
        else:
            return base_lr
    
    def _secure_aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Secure aggregation (placeholder for test utility method).
        
        Args:
            client_updates: Client parameter updates
            
        Returns:
            Securely aggregated parameters
        """
        # For testing, just return regular aggregation
        return self._aggregate_parameters(client_updates)
    
    def get_convergence_metrics(self) -> Dict[str, List[float]]:
        """
        Get convergence tracking metrics (test utility method).
        
        Returns:
            Convergence metrics
        """
        if not hasattr(self, 'convergence_metrics'):
            self.convergence_metrics = {
                'parameter_changes': [],
                'gradient_norms': []
            }
        return self.convergence_metrics
    
    def _update_convergence_metrics(
        self, 
        old_params: Dict[str, torch.Tensor], 
        new_params: Dict[str, torch.Tensor], 
        round_num: int
    ):
        """
        Update convergence metrics (test utility method).
        
        Args:
            old_params: Previous parameters
            new_params: New parameters
            round_num: Round number
        """
        if not hasattr(self, 'convergence_metrics'):
            self.convergence_metrics = {
                'parameter_changes': [],
                'gradient_norms': []
            }
        
        # Calculate parameter change
        if old_params and new_params:
            total_change = 0.0
            total_norm = 0.0
            for key in old_params.keys():
                if key in new_params:
                    change = torch.norm(new_params[key] - old_params[key])
                    norm = torch.norm(new_params[key])
                    total_change += change.item()
                    total_norm += norm.item()
            
            self.convergence_metrics['parameter_changes'].append(total_change)
            self.convergence_metrics['gradient_norms'].append(total_norm)
    
    @staticmethod
    def _aggregate_parameters_static(client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Static version of parameter aggregation for testing.
        """
        if not client_updates:
            return {}
        
        aggregated = {}
        for key in client_updates[0].keys():
            params = [update[key] for update in client_updates if key in update]
            if params:
                aggregated[key] = torch.stack(params).mean(dim=0)
        
        return aggregated
    
    @staticmethod
    def _aggregate_parameters_weighted_static(
        client_updates: List[Dict[str, torch.Tensor]], 
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Static version of weighted parameter aggregation for testing.
        """
        if not client_updates or not client_weights or len(client_updates) != len(client_weights):
            return {}
        
        total_weight = sum(client_weights)
        if total_weight == 0:
            return SecureFedAvgStrategy._aggregate_parameters_static(client_updates)
        
        aggregated = {}
        for key in client_updates[0].keys():
            weighted_sum = torch.zeros_like(client_updates[0][key])
            for update, weight in zip(client_updates, client_weights):
                if key in update:
                    weighted_sum += update[key] * weight
            
            aggregated[key] = weighted_sum / total_weight
        
        return aggregated


class DifferentialPrivacyStrategy(SecureFedAvgStrategy):
    """
    FedAvg strategy with additional server-side differential privacy.
    """
    
    def __init__(
        self,
        server_noise_multiplier: float = 0.1,
        server_max_norm: float = 1.0,
        **kwargs
    ):
        """
        Initialize DP strategy.
        
        Args:
            server_noise_multiplier: Noise multiplier for server-side DP
            server_max_norm: Max norm for clipping
            **kwargs: Arguments for SecureFedAvgStrategy
        """
        super().__init__(**kwargs)
        self.server_noise_multiplier = server_noise_multiplier
        self.server_max_norm = server_max_norm
    
    def _fedavg_aggregate(
        self,
        parameters_list: List[List[np.ndarray]],
        num_examples_list: List[int]
    ) -> List[np.ndarray]:
        """
        Perform FedAvg with server-side DP noise.
        
        Args:
            parameters_list: List of parameter arrays
            num_examples_list: List of sample counts
            
        Returns:
            Aggregated parameters with DP noise
        """
        # Get base aggregation
        aggregated = super()._fedavg_aggregate(parameters_list, num_examples_list)
        
        # Add Gaussian noise for DP
        noisy_aggregated = []
        for param in aggregated:
            # Clip parameter norm
            param_norm = np.linalg.norm(param)
            if param_norm > self.server_max_norm:
                param = param * (self.server_max_norm / param_norm)
            
            # Add Gaussian noise
            noise = np.random.normal(
                0.0,
                self.server_noise_multiplier * self.server_max_norm,
                param.shape
            )
            
            noisy_param = param + noise
            noisy_aggregated.append(noisy_param)
        
        return noisy_aggregated
    
    def _aggregate_parameters(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate parameters from multiple clients (test utility method).
        
        Args:
            client_updates: List of parameter dictionaries from clients
            
        Returns:
            Aggregated parameters
        """
        if not client_updates:
            return {}
        
        # Simple average aggregation
        aggregated = {}
        for key in client_updates[0].keys():
            params = [update[key] for update in client_updates if key in update]
            if params:
                aggregated[key] = torch.stack(params).mean(dim=0)
        
        return aggregated
    
    def _aggregate_parameters_weighted(
        self, 
        client_updates: List[Dict[str, torch.Tensor]], 
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Weighted aggregation of parameters (test utility method).
        
        Args:
            client_updates: List of parameter dictionaries
            client_weights: Weights for each client
            
        Returns:
            Weighted aggregated parameters
        """
        if not client_updates or not client_weights or len(client_updates) != len(client_weights):
            return {}
        
        total_weight = sum(client_weights)
        if total_weight == 0:
            return self._aggregate_parameters(client_updates)
        
        aggregated = {}
        for key in client_updates[0].keys():
            weighted_sum = torch.zeros_like(client_updates[0][key])
            for update, weight in zip(client_updates, client_weights):
                if key in update:
                    weighted_sum += update[key] * weight
            
            aggregated[key] = weighted_sum / total_weight
        
        return aggregated
    
    def _robust_aggregate_parameters(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Robust aggregation with outlier removal (test utility method).
        
        Args:
            client_updates: List of parameter dictionaries
            
        Returns:
            Robustly aggregated parameters
        """
        # Simple implementation: remove outliers based on norm
        if len(client_updates) <= 2:
            return self._aggregate_parameters(client_updates)
        
        # Calculate norms for each update
        norms = []
        for update in client_updates:
            total_norm = sum(torch.norm(param) ** 2 for param in update.values()) ** 0.5
            norms.append(total_norm.item())
        
        # Remove outliers (simple: remove highest and lowest)
        sorted_indices = sorted(range(len(norms)), key=lambda i: norms[i])
        keep_indices = sorted_indices[1:-1]  # Remove highest and lowest
        
        filtered_updates = [client_updates[i] for i in keep_indices]
        return self._aggregate_parameters(filtered_updates)
    
    def _aggregate_evaluation_metrics(self, client_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate evaluation metrics from clients (test utility method).
        
        Args:
            client_results: List of metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        if not client_results:
            return {}
        
        aggregated = {}
        keys = client_results[0].keys()
        
        for key in keys:
            values = [result.get(key, 0.0) for result in client_results if key in result]
            if values:
                aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    def _select_clients(self, available_clients: List[str], fraction: float = 1.0, min_clients: int = 1) -> List[str]:
        """
        Select subset of clients for training (test utility method).
        
        Args:
            available_clients: List of available client IDs
            fraction: Fraction to select
            min_clients: Minimum number to select
            
        Returns:
            Selected client IDs
        """
        num_to_select = max(min_clients, int(len(available_clients) * fraction))
        num_to_select = min(num_to_select, len(available_clients))
        
        # Simple random selection
        import random
        return random.sample(available_clients, num_to_select)
    
    def _validate_parameters(self, parameters: Dict[str, torch.Tensor]) -> bool:
        """
        Validate parameter dictionary (test utility method).
        
        Args:
            parameters: Parameter dictionary to validate
            
        Returns:
            True if valid
        """
        if not parameters:
            return False
        
        for key, param in parameters.items():
            if param is None or torch.isnan(param).any() or torch.isinf(param).any():
                return False
        
        return True
    
    def _clip_gradients(self, gradients: Dict[str, torch.Tensor], max_norm: float) -> Dict[str, torch.Tensor]:
        """
        Clip gradients by global norm (test utility method).
        
        Args:
            gradients: Gradient dictionary
            max_norm: Maximum norm
            
        Returns:
            Clipped gradients
        """
        # Calculate total norm
        total_norm = sum(torch.norm(grad) ** 2 for grad in gradients.values()) ** 0.5
        
        if total_norm > max_norm:
            scale = max_norm / total_norm
            return {key: grad * scale for key, grad in gradients.items()}
        
        return gradients
    
    def _compute_adaptive_learning_rate(self, base_lr: float) -> float:
        """
        Compute adaptive learning rate based on convergence (test utility method).
        
        Args:
            base_lr: Base learning rate
            
        Returns:
            Adaptive learning rate
        """
        if not hasattr(self, 'convergence_metrics') or not self.convergence_metrics.get('parameter_changes'):
            return base_lr
        
        changes = self.convergence_metrics['parameter_changes']
        recent_change = changes[-1] if changes else 0.0
        
        # Increase LR if convergence is slow, decrease if fast
        if recent_change > 0.01:  # Slow convergence
            return base_lr * 1.2
        elif recent_change < 0.001:  # Fast convergence
            return base_lr * 0.8
        else:
            return base_lr
    
    def _secure_aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Secure aggregation (placeholder for test utility method).
        
        Args:
            client_updates: Client parameter updates
            
        Returns:
            Securely aggregated parameters
        """
        # For testing, just return regular aggregation
        return self._aggregate_parameters(client_updates)
    
    def get_convergence_metrics(self) -> Dict[str, List[float]]:
        """
        Get convergence tracking metrics (test utility method).
        
        Returns:
            Convergence metrics
        """
        if not hasattr(self, 'convergence_metrics'):
            self.convergence_metrics = {
                'parameter_changes': [],
                'gradient_norms': []
            }
        return self.convergence_metrics
    
    def _update_convergence_metrics(
        self, 
        old_params: Dict[str, torch.Tensor], 
        new_params: Dict[str, torch.Tensor], 
        round_num: int
    ):
        """
        Update convergence metrics (test utility method).
        
        Args:
            old_params: Previous parameters
            new_params: New parameters
            round_num: Round number
        """
        if not hasattr(self, 'convergence_metrics'):
            self.convergence_metrics = {
                'parameter_changes': [],
                'gradient_norms': []
            }
        
        # Calculate parameter change
        if old_params and new_params:
            total_change = 0.0
            total_norm = 0.0
            for key in old_params.keys():
                if key in new_params:
                    change = torch.norm(new_params[key] - old_params[key])
                    norm = torch.norm(new_params[key])
                    total_change += change.item()
                    total_norm += norm.item()
            
            self.convergence_metrics['parameter_changes'].append(total_change)
            self.convergence_metrics['gradient_norms'].append(total_norm)
    
    @staticmethod
    def _aggregate_parameters_static(client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Static version of parameter aggregation for testing.
        """
        if not client_updates:
            return {}
        
        aggregated = {}
        for key in client_updates[0].keys():
            params = [update[key] for update in client_updates if key in update]
            if params:
                aggregated[key] = torch.stack(params).mean(dim=0)
        
        return aggregated
    
    @staticmethod
    def _aggregate_parameters_weighted_static(
        client_updates: List[Dict[str, torch.Tensor]], 
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Static version of weighted parameter aggregation for testing.
        """
        if not client_updates or not client_weights or len(client_updates) != len(client_weights):
            return {}
        
        total_weight = sum(client_weights)
        if total_weight == 0:
            return SecureFedAvgStrategy._aggregate_parameters_static(client_updates)
        
        aggregated = {}
        for key in client_updates[0].keys():
            weighted_sum = torch.zeros_like(client_updates[0][key])
            for update, weight in zip(client_updates, client_weights):
                if key in update:
                    weighted_sum += update[key] * weight
            
            aggregated[key] = weighted_sum / total_weight
        
        return aggregated
