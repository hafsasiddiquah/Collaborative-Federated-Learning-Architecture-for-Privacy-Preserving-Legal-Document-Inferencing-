"""
Inference Engine for Legal Document Summarization
Handles model loading, inference, and resource management.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

# Try to import PEFT, but fall back gracefully
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available, using base model loading")

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Handles model inference for legal document summarization.
    Manages model loading, memory optimization, and inference.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_memory: Optional[Dict] = None,
        use_fp16: bool = True
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to the model directory
            device: Device to load model on ("auto", "cuda", "cpu")
            max_memory: Maximum memory per GPU (for multi-GPU)
            use_fp16: Whether to use FP16 precision
        """
        self.model_path = model_path
        self.device = device
        self.max_memory = max_memory or {0: "4GB", "cpu": "8GB"}  # Default memory limits
        self.use_fp16 = use_fp16

        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.is_loaded = False

        # Load model on initialization
        self.load_model()

    def load_model(self) -> bool:
        """
        Load the model and tokenizer.

        Returns:
            True if loading successful, False otherwise
        """
        try:
            logger.info(f"Loading model from {self.model_path}")

            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Configure quantization for memory efficiency
            quantization_config = None
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )

            # Load tokenizer
            tokenizer_path = os.path.join(self.model_path, "tokenizer")
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                # Try loading from model path
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Load base model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.use_fp16 else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "max_memory": self.max_memory if self.device == "cuda" else None
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            # Try loading as PEFT model first
            adapter_path = os.path.join(self.model_path, "adapter_model")
            if os.path.exists(adapter_path) and PEFT_AVAILABLE:
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    "google/pegasus-xsum",
                    **model_kwargs
                )

                # Load PEFT adapter
                self.model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path
                )
                logger.info("Loaded PEFT model with adapter")
            elif os.path.exists(adapter_path) and not PEFT_AVAILABLE:
                logger.warning("PEFT adapter found but PEFT not available, loading base model only")
                # Load full model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            else:
                # Load full model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )

            # Move to device if not using device_map
            if self.device != "cuda" or not quantization_config:
                self.model.to(self.device)

            self.model.eval()
            self.is_loaded = True

            logger.info(f"Model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False

    def generate_summary(
        self,
        text: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 4,
        do_sample: bool = True,
        repetition_penalty: float = 1.1
    ) -> Dict[str, Any]:
        """
        Generate summary for input text.

        Args:
            text: Input document text
            max_length: Maximum summary length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty

        Returns:
            Dictionary with summary and metadata
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        try:
            # Prepare input
            prompt = f"Summarize the following legal document:\n\n{text}\n\nSummary:"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )

            # Decode
            summary = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Clean up summary (remove prompt if present)
            if "Summary:" in summary:
                summary = summary.split("Summary:")[-1].strip()

            return {
                "summary": summary,
                "input_length": len(text),
                "summary_length": len(summary),
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_beams": num_beams
                }
            }

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Model information dictionary
        """
        if not self.is_loaded:
            return {"status": "not_loaded"}

        info = {
            "status": "loaded",
            "device": str(self.device),
            "model_path": self.model_path,
            "use_fp16": self.use_fp16
        }

        if hasattr(self.model, "config"):
            config = self.model.config
            info.update({
                "model_type": config.model_type,
                "vocab_size": config.vocab_size,
                "max_position_embeddings": getattr(config, "max_position_embeddings", None),
                "hidden_size": getattr(config, "hidden_size", None),
                "num_attention_heads": getattr(config, "num_attention_heads", None),
                "num_hidden_layers": getattr(config, "num_hidden_layers", None)
            })

        # Get model size
        if hasattr(self.model, "parameters"):
            param_count = sum(p.numel() for p in self.model.parameters())
            info["parameter_count"] = param_count

        return info

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_loaded = False
        logger.info("Model unloaded")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload_model()
        except AttributeError:
            pass  # Model was never loaded