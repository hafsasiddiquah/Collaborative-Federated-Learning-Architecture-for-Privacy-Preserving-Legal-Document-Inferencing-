"""
AES-256 Encryption Module for Secure Gradient Transmission
Encrypts model gradients before sending to aggregation server.
"""

import os
import json
import pickle
import base64
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets


class AES256Encryption:
    """
    AES-256 encryption for securing model gradients and parameters.
    Uses CBC mode with PKCS7 padding for compatibility.
    """
    
    def __init__(
        self,
        key: Optional[bytes] = None,
        key_derivation_salt: Optional[bytes] = None
    ):
        """
        Initialize AES-256 encryption engine.
        
        Args:
            key: 32-byte encryption key (if None, generates random key)
            key_derivation_salt: Salt for key derivation (if None, generates random)
        """
        self.key_size = 32  # 256 bits
        self.block_size = 16  # AES block size
        
        if key is None:
            # Generate random key
            self.key = secrets.token_bytes(self.key_size)
        else:
            if len(key) != self.key_size:
                raise ValueError(f"Key must be {self.key_size} bytes for AES-256")
            self.key = key
        
        if key_derivation_salt is None:
            self.salt = secrets.token_bytes(16)
        else:
            self.salt = key_derivation_salt
    
    @staticmethod
    def derive_key_from_password(
        password: str,
        salt: bytes,
        iterations: int = 100000
    ) -> bytes:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: Password string
            salt: Salt bytes
            iterations: PBKDF2 iterations
            
        Returns:
            Derived 32-byte key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        key = kdf.derive(password.encode('utf-8'))
        return key
    
    def encrypt_tensor_dict(
        self,
        data: Dict[str, Any],
        iv: Optional[bytes] = None
    ) -> Dict[str, str]:
        """
        Encrypt a dictionary of tensors/parameters.
        
        Args:
            data: Dictionary to encrypt (can contain torch tensors)
            iv: Initialization vector (if None, generates random)
            
        Returns:
            Dictionary with encrypted data and metadata
        """
        if iv is None:
            iv = secrets.token_bytes(self.block_size)
        
        # Serialize data to bytes
        serialized_data = pickle.dumps(data)
        
        # Pad data to block size
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(serialized_data)
        padded_data += padder.finalize()
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encrypt
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Encode to base64 for transmission
        encrypted_b64 = base64.b64encode(ciphertext).decode('utf-8')
        iv_b64 = base64.b64encode(iv).decode('utf-8')
        salt_b64 = base64.b64encode(self.salt).decode('utf-8')
        
        return {
            "encrypted_data": encrypted_b64,
            "iv": iv_b64,
            "salt": salt_b64,
            "algorithm": "AES-256-CBC"
        }
    
    def decrypt_tensor_dict(
        self,
        encrypted_dict: Dict[str, str],
        key: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Decrypt a dictionary of tensors/parameters.
        
        Args:
            encrypted_dict: Dictionary with encrypted data and metadata
            key: Decryption key (if None, uses self.key)
            
        Returns:
            Decrypted dictionary
        """
        if key is None:
            key = self.key
        
        # Decode from base64
        ciphertext = base64.b64decode(encrypted_dict["encrypted_data"])
        iv = base64.b64decode(encrypted_dict["iv"])
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Unpad
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data)
        data += unpadder.finalize()
        
        # Deserialize
        decrypted_dict = pickle.loads(data)
        
        return decrypted_dict
    
    def encrypt_gradients(
        self,
        gradients: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Encrypt model gradients for secure transmission.
        
        Args:
            gradients: Dictionary of gradient tensors
            
        Returns:
            Encrypted gradients dictionary
        """
        return self.encrypt_tensor_dict(gradients)
    
    def decrypt_gradients(
        self,
        encrypted_gradients: Dict[str, str],
        key: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Decrypt model gradients received from client.
        
        Args:
            encrypted_gradients: Encrypted gradients dictionary
            key: Decryption key
            
        Returns:
            Decrypted gradients dictionary
        """
        return self.decrypt_tensor_dict(encrypted_gradients, key)
    
    def encrypt_parameters(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Encrypt model parameters for secure transmission.
        
        Args:
            parameters: Dictionary of parameter tensors
            
        Returns:
            Encrypted parameters dictionary
        """
        return self.encrypt_tensor_dict(parameters)
    
    def decrypt_parameters(
        self,
        encrypted_parameters: Dict[str, str],
        key: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Decrypt model parameters received from server.
        
        Args:
            encrypted_parameters: Encrypted parameters dictionary
            key: Decryption key
            
        Returns:
            Decrypted parameters dictionary
        """
        return self.decrypt_tensor_dict(encrypted_parameters, key)
    
    def save_key(self, filepath: str):
        """
        Save encryption key to file (for key exchange).
        
        Args:
            filepath: Path to save key file
        """
        key_data = {
            "key": base64.b64encode(self.key).decode('utf-8'),
            "salt": base64.b64encode(self.salt).decode('utf-8')
        }
        
        with open(filepath, 'w') as f:
            json.dump(key_data, f)
        
        print(f"Key saved to {filepath}")
    
    def load_key(self, filepath: str):
        """
        Load encryption key from file.
        
        Args:
            filepath: Path to key file
        """
        with open(filepath, 'r') as f:
            key_data = json.load(f)
        
        self.key = base64.b64decode(key_data["key"])
        self.salt = base64.b64decode(key_data["salt"])
        
        print(f"Key loaded from {filepath}")
    
    def get_key(self) -> bytes:
        """
        Get current encryption key.
        
        Returns:
            Encryption key bytes
        """
        return self.key
    
    def set_key(self, key: bytes):
        """
        Set encryption key.
        
        Args:
            key: 32-byte encryption key
        """
        if len(key) != self.key_size:
            raise ValueError(f"Key must be {self.key_size} bytes for AES-256")
        self.key = key
