"""
Unit Tests for Encryption Module
Tests AES-256 encryption/decryption integrity and security properties.
"""

import torch
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from client.encryption import AES256Encryption


class TestAES256Encryption:
    """Test cases for AES-256 encryption."""

    def setup_method(self):
        """Setup test fixtures."""
        self.encryption = AES256Encryption()
        self.test_data = {
            "param1": torch.randn(100, 50),
            "param2": torch.randn(200, 100),
            "metadata": {"epoch": 5, "loss": 0.123}
        }

    def test_initialization(self):
        """Test encryption engine initialization."""
        enc = AES256Encryption()

        assert enc.key_size == 32
        assert enc.block_size == 16
        assert len(enc.key) == 32
        assert len(enc.salt) == 16

    def test_initialization_with_key(self):
        """Test initialization with provided key."""
        custom_key = b"A" * 32
        custom_salt = b"B" * 16

        enc = AES256Encryption(key=custom_key, key_derivation_salt=custom_salt)

        assert enc.key == custom_key
        assert enc.salt == custom_salt

    def test_invalid_key_size(self):
        """Test that invalid key sizes are rejected."""
        with pytest.raises(ValueError, match="Key must be 32 bytes"):
            AES256Encryption(key=b"short_key")

    def test_key_derivation(self):
        """Test PBKDF2 key derivation."""
        password = "test_password"
        salt = b"salt123456789012"

        derived_key = AES256Encryption.derive_key_from_password(password, salt)

        assert len(derived_key) == 32
        assert isinstance(derived_key, bytes)

        # Same password and salt should produce same key
        derived_key2 = AES256Encryption.derive_key_from_password(password, salt)
        assert derived_key == derived_key2

        # Different salt should produce different key
        different_salt = b"different_salt_12"
        derived_key3 = AES256Encryption.derive_key_from_password(password, different_salt)
        assert derived_key != derived_key3

    def test_tensor_dict_encryption_decryption(self):
        """Test encryption and decryption of tensor dictionaries."""
        # Encrypt
        encrypted = self.encryption.encrypt_tensor_dict(self.test_data)

        assert "encrypted_data" in encrypted
        assert "iv" in encrypted
        assert "salt" in encrypted
        assert encrypted["algorithm"] == "AES-256-CBC"

        # Decrypt
        decrypted = self.encryption.decrypt_tensor_dict(encrypted)

        # Check that data is recovered correctly
        assert len(decrypted) == len(self.test_data)

        for key in self.test_data:
            if isinstance(self.test_data[key], torch.Tensor):
                assert torch.allclose(decrypted[key], self.test_data[key])
            else:
                assert decrypted[key] == self.test_data[key]

    def test_encryption_with_custom_iv(self):
        """Test encryption with custom initialization vector."""
        custom_iv = b"C" * 16

        encrypted = self.encryption.encrypt_tensor_dict(self.test_data, iv=custom_iv)

        assert encrypted["iv"] == "QysrKywtLisrKywtLisrKywtLisrKywt"  # base64 of custom_iv

    def test_decryption_with_different_key(self):
        """Test that decryption fails with wrong key."""
        # Encrypt with one instance
        enc1 = AES256Encryption()
        encrypted = enc1.encrypt_tensor_dict(self.test_data)

        # Try to decrypt with different instance (different key)
        enc2 = AES256Encryption()

        with pytest.raises(Exception):  # Should fail due to wrong key
            enc2.decrypt_tensor_dict(encrypted)

    def test_gradient_encryption_methods(self):
        """Test gradient-specific encryption methods."""
        gradients = {
            "layer1.weight": torch.randn(50, 100),
            "layer1.bias": torch.randn(50),
            "layer2.weight": torch.randn(10, 50)
        }

        # Encrypt gradients
        encrypted = self.encryption.encrypt_gradients(gradients)

        assert "encrypted_data" in encrypted

        # Decrypt gradients
        decrypted = self.encryption.decrypt_gradients(encrypted)

        # Verify correctness
        for name, grad in gradients.items():
            assert torch.allclose(decrypted[name], grad)

    def test_parameter_encryption_methods(self):
        """Test parameter-specific encryption methods."""
        parameters = {
            "layer1.weight": torch.randn(50, 100),
            "layer1.bias": torch.randn(50),
            "layer2.weight": torch.randn(10, 50)
        }

        # Encrypt parameters
        encrypted = self.encryption.encrypt_parameters(parameters)

        assert "encrypted_data" in encrypted

        # Decrypt parameters
        decrypted = self.encryption.decrypt_parameters(encrypted)

        # Verify correctness
        for name, param in parameters.items():
            assert torch.allclose(decrypted[name], param)

    def test_key_save_load(self):
        """Test saving and loading encryption keys."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            key_file = tmp_file.name

        try:
            # Save key
            self.encryption.save_key(key_file)

            # Create new instance and load key
            new_encryption = AES256Encryption()
            new_encryption.load_key(key_file)

            # Test that they can decrypt each other's data
            encrypted = self.encryption.encrypt_tensor_dict(self.test_data)
            decrypted = new_encryption.decrypt_tensor_dict(encrypted)

            # Verify data integrity
            for key in self.test_data:
                if isinstance(self.test_data[key], torch.Tensor):
                    assert torch.allclose(decrypted[key], self.test_data[key])
                else:
                    assert decrypted[key] == self.test_data[key]

        finally:
            os.unlink(key_file)

    def test_get_set_key(self):
        """Test key getter and setter."""
        original_key = self.encryption.get_key()
        assert len(original_key) == 32

        new_key = b"D" * 32
        self.encryption.set_key(new_key)
        assert self.encryption.get_key() == new_key

    def test_set_invalid_key(self):
        """Test that invalid key sizes are rejected."""
        with pytest.raises(ValueError, match="Key must be 32 bytes"):
            self.encryption.set_key(b"invalid")

    def test_large_data_encryption(self):
        """Test encryption of large data structures."""
        # Create large tensor data
        large_data = {
            "large_tensor": torch.randn(1000, 1000),
            "metadata": {"size": "large", "type": "test"}
        }

        # Encrypt
        encrypted = self.encryption.encrypt_tensor_dict(large_data)

        # Decrypt
        decrypted = self.encryption.decrypt_tensor_dict(encrypted)

        # Verify
        assert torch.allclose(decrypted["large_tensor"], large_data["large_tensor"])
        assert decrypted["metadata"] == large_data["metadata"]

    def test_empty_data_encryption(self):
        """Test encryption of empty data."""
        empty_data = {}

        encrypted = self.encryption.encrypt_tensor_dict(empty_data)
        decrypted = self.encryption.decrypt_tensor_dict(encrypted)

        assert decrypted == empty_data

    def test_mixed_data_types(self):
        """Test encryption of mixed data types."""
        mixed_data = {
            "tensor_float": torch.randn(10, 10),
            "tensor_int": torch.randint(0, 100, (5, 5)),
            "numpy_array": np.random.randn(5, 5),
            "string": "test_string",
            "number": 42,
            "list": [1, 2, 3, "four"],
            "dict": {"nested": "value", "number": 123}
        }

        # Encrypt
        encrypted = self.encryption.encrypt_tensor_dict(mixed_data)

        # Decrypt
        decrypted = self.encryption.decrypt_tensor_dict(encrypted)

        # Verify each type
        assert torch.allclose(decrypted["tensor_float"], mixed_data["tensor_float"])
        assert torch.equal(decrypted["tensor_int"], mixed_data["tensor_int"])
        assert np.allclose(decrypted["numpy_array"], mixed_data["numpy_array"])
        assert decrypted["string"] == mixed_data["string"]
        assert decrypted["number"] == mixed_data["number"]
        assert decrypted["list"] == mixed_data["list"]
        assert decrypted["dict"] == mixed_data["dict"]


class TestEncryptionSecurity:
    """Test encryption security properties."""

    def test_ciphertext_uniqueness(self):
        """Test that same plaintext produces different ciphertext."""
        data = {"test": torch.ones(10, 10)}

        enc1 = AES256Encryption()
        enc2 = AES256Encryption()

        encrypted1 = enc1.encrypt_tensor_dict(data)
        encrypted2 = enc2.encrypt_tensor_dict(data)

        # Different keys should produce different ciphertext
        assert encrypted1["encrypted_data"] != encrypted2["encrypted_data"]

        # Different IVs should produce different ciphertext (even with same key)
        encrypted3 = enc1.encrypt_tensor_dict(data)
        assert encrypted1["encrypted_data"] != encrypted3["encrypted_data"]

    def test_ciphertext_randomness(self):
        """Test that ciphertext appears random."""
        data = {"test": torch.ones(100, 100)}

        encrypted_results = []
        for _ in range(10):
            encrypted = self.encryption.encrypt_tensor_dict(data)
            encrypted_results.append(encrypted["encrypted_data"])

        # All ciphertexts should be different
        for i in range(len(encrypted_results)):
            for j in range(i+1, len(encrypted_results)):
                assert encrypted_results[i] != encrypted_results[j]

    def test_corruption_detection(self):
        """Test that corrupted ciphertext is detected."""
        encrypted = self.encryption.encrypt_tensor_dict(self.test_data)

        # Corrupt the ciphertext
        corrupted = encrypted.copy()
        corrupted_data = bytearray(corrupted["encrypted_data"], 'utf-8')
        corrupted_data[10] = corrupted_data[10] ^ 0xFF  # Flip a bit
        corrupted["encrypted_data"] = corrupted_data.decode('utf-8')

        # Decryption should fail
        with pytest.raises(Exception):
            self.encryption.decrypt_tensor_dict(corrupted)


class TestEncryptionIntegration:
    """Integration tests for encryption in federated learning context."""

    def test_federated_learning_simulation(self):
        """Simulate encryption in federated learning workflow."""
        # Simulate two clients
        client1_enc = AES256Encryption()
        client2_enc = AES256Encryption()

        # Client 1 encrypts gradients
        client1_gradients = {
            "layer1": torch.randn(50, 100),
            "layer2": torch.randn(10, 50)
        }
        client1_encrypted = client1_enc.encrypt_gradients(client1_gradients)

        # Client 2 encrypts gradients
        client2_gradients = {
            "layer1": torch.randn(50, 100),
            "layer2": torch.randn(10, 50)
        }
        client2_encrypted = client2_enc.encrypt_gradients(client2_gradients)

        # Server decrypts (in practice, would need shared keys)
        # For testing, server uses same key as client
        server_enc = AES256Encryption(key=client1_enc.get_key())

        client1_decrypted = server_enc.decrypt_gradients(client1_encrypted)
        client2_decrypted = server_enc.decrypt_gradients(client2_encrypted)

        # Verify correctness
        for name in client1_gradients:
            assert torch.allclose(client1_decrypted[name], client1_gradients[name])

        for name in client2_gradients:
            assert torch.allclose(client2_decrypted[name], client2_gradients[name])


if __name__ == "__main__":
    pytest.main([__file__])