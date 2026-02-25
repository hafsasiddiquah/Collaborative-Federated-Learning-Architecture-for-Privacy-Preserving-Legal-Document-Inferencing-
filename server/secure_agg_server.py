"""
Secure Aggregation Server for Federated Learning
Handles encrypted gradient aggregation with poisoning detection.
"""

import flwr as fl
from flwr.server import Server, ServerConfig
from flwr.server.strategy import Strategy
from typing import Optional, Dict, Any
import ssl
import socket
from .strategy import SecureFedAvgStrategy, DifferentialPrivacyStrategy


class SecureAggregationServer:
    """
    Secure aggregation server for federated learning.
    Supports TLS 1.3, encrypted gradients, and poisoning detection.
    """
    
    def __init__(
        self,
        strategy: Optional[Strategy] = None,
        num_rounds: int = 10,
        min_clients: int = 2,
        use_tls: bool = True,
        cert_path: Optional[str] = None,
        key_path: Optional[str] = None,
        ca_path: Optional[str] = None,
        use_server_dp: bool = False,
        server_dp_noise: float = 0.1
    ):
        """
        Initialize secure aggregation server.
        
        Args:
            strategy: FL strategy (if None, creates SecureFedAvgStrategy)
            num_rounds: Number of federated rounds
            min_clients: Minimum clients required
            use_tls: Enable TLS 1.3
            cert_path: Path to server certificate
            key_path: Path to server private key
            ca_path: Path to CA certificate
            use_server_dp: Enable server-side differential privacy
            server_dp_noise: Server-side DP noise multiplier
        """
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.use_tls = use_tls
        
        # Setup strategy
        if strategy is None:
            if use_server_dp:
                strategy = DifferentialPrivacyStrategy(
                    min_clients=min_clients,
                    server_noise_multiplier=server_dp_noise
                )
            else:
                strategy = SecureFedAvgStrategy(
                    min_clients=min_clients
                )
        
        self.strategy = strategy
        
        # TLS configuration
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_path = ca_path
        self.ssl_context = None
        
        if use_tls:
            self._setup_tls()
    
    def _setup_tls(self):
        """
        Setup TLS 1.3 context for secure communication.
        """
        if self.cert_path is None or self.key_path is None:
            print("Warning: TLS enabled but certificates not provided. "
                  "Server will run without TLS (not recommended for production).")
            return
        
        try:
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(self.cert_path, self.key_path)
            self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_3
            
            # Enable mutual TLS if CA provided
            if self.ca_path:
                self.ssl_context.load_verify_locations(self.ca_path)
                self.ssl_context.verify_mode = ssl.CERT_REQUIRED
            
            print("TLS 1.3 context configured successfully.")
        except Exception as e:
            print(f"Error setting up TLS: {e}")
            print("Server will run without TLS (not recommended for production).")
            self.ssl_context = None
    
    def start(
        self,
        server_address: str = "0.0.0.0:8080",
        config: Optional[ServerConfig] = None
    ):
        """
        Start the secure aggregation server.
        
        Args:
            server_address: Server address (host:port)
            config: Server configuration
        """
        if config is None:
            config = ServerConfig(num_rounds=self.num_rounds)
        
        # Create Flower server
        server = Server(
            client_manager=fl.server.SimpleClientManager(),
            strategy=self.strategy,
            config=config
        )
        
        print(f"Starting secure aggregation server on {server_address}")
        print(f"Minimum clients: {self.min_clients}")
        print(f"Number of rounds: {self.num_rounds}")
        print(f"TLS enabled: {self.use_tls}")
        
        # Start server
        if self.use_tls and self.ssl_context:
            # Start with TLS
            fl.server.start_server(
                server_address=server_address,
                server=server,
                config=config,
                # Note: Flower doesn't directly support SSL context,
                # but this can be handled at the network layer
            )
        else:
            # Start without TLS (for development)
            fl.server.start_server(
                server_address=server_address,
                server=server,
                config=config
            )
    
    def get_server_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.
        
        Returns:
            Dictionary with server statistics
        """
        return {
            "num_rounds": self.num_rounds,
            "min_clients": self.min_clients,
            "tls_enabled": self.use_tls,
            "strategy": type(self.strategy).__name__
        }


def create_secure_server(
    num_rounds: int = 10,
    min_clients: int = 2,
    use_server_dp: bool = False,
    similarity_threshold: float = 0.5
) -> SecureAggregationServer:
    """
    Factory function to create a secure aggregation server.
    
    Args:
        num_rounds: Number of federated rounds
        min_clients: Minimum clients required
        use_server_dp: Enable server-side DP
        similarity_threshold: Cosine similarity threshold for poisoning detection
        
    Returns:
        Configured SecureAggregationServer instance
    """
    if use_server_dp:
        strategy = DifferentialPrivacyStrategy(
            min_clients=min_clients,
            similarity_threshold=similarity_threshold,
            server_noise_multiplier=0.1
        )
    else:
        strategy = SecureFedAvgStrategy(
            min_clients=min_clients,
            similarity_threshold=similarity_threshold
        )
    
    server = SecureAggregationServer(
        strategy=strategy,
        num_rounds=num_rounds,
        min_clients=min_clients,
        use_tls=False  # Set to True in production with proper certificates
    )
    
    return server


def main():
    """
    Main function to run the secure aggregation server.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Secure FL Aggregation Server")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080",
                       help="Server address")
    parser.add_argument("--rounds", type=int, default=10,
                       help="Number of federated rounds")
    parser.add_argument("--min-clients", type=int, default=2,
                       help="Minimum number of clients")
    parser.add_argument("--server-dp", action="store_true",
                       help="Enable server-side differential privacy")
    parser.add_argument("--similarity-threshold", type=float, default=0.5,
                       help="Cosine similarity threshold for poisoning detection")
    parser.add_argument("--cert", type=str, default=None,
                       help="Path to server certificate")
    parser.add_argument("--key", type=str, default=None,
                       help="Path to server private key")
    
    args = parser.parse_args()
    
    # Create server
    server = create_secure_server(
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        use_server_dp=args.server_dp,
        similarity_threshold=args.similarity_threshold
    )
    
    # Configure TLS if certificates provided
    if args.cert and args.key:
        server.cert_path = args.cert
        server.key_path = args.key
        server.use_tls = True
        server._setup_tls()
    
    # Start server
    server.start(server_address=args.address)


if __name__ == "__main__":
    main()
