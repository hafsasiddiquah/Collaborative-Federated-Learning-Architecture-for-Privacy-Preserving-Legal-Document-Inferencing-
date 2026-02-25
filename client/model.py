"""
LLaMA Model with QLoRA (4-bit Quantization + LoRA Adapters)
for Legal Document Summarization
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Optional, Dict, Any


class QLoRALegalModel:
    """
    Transformer model with QLoRA for efficient fine-tuning on legal documents.
    Uses 4-bit quantization and LoRA adapters for parameter-efficient training.
    """
    
    def __init__(
        self,
        model_name: str = "google/pegasus-xsum",
        cache_dir: str = "./model_cache",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None,
        use_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True
    ):
        """
        Initialize QLoRA model for legal document summarization.
        
        Args:
            model_name: Base LLaMA model name
            cache_dir: Directory to cache models
            lora_r: LoRA rank (number of trainable parameters)
            lora_alpha: LoRA alpha scaling parameter
            lora_dropout: LoRA dropout rate
            target_modules: Modules to apply LoRA to (default: q_proj, v_proj)
            use_4bit: Enable 4-bit quantization
            bnb_4bit_compute_dtype: Compute dtype for 4-bit
            bnb_4bit_quant_type: Quantization type (nf4 or fp4)
            bnb_4bit_use_double_quant: Use double quantization
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Default target modules for LLaMA
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.target_modules = target_modules
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure 4-bit quantization
        if use_4bit:
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant
            )
        else:
            self.quantization_config = None
        
        # Load base model
        self.model = None
        self.peft_model = None
        
    def load_model(self) -> nn.Module:
        """
        Load and configure the base LLaMA model with QLoRA.
        
        Returns:
            Configured PEFT model ready for training
        """
        print(f"Loading base model: {self.model_name}")
        
        # Load model with quantization if enabled
        model_kwargs = {
            "trust_remote_code": True,
            "cache_dir": self.cache_dir,
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        if self.quantization_config:
            model_kwargs["quantization_config"] = self.quantization_config
        
        # Try Seq2Seq first, fallback to CausalLM
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        except:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        
        # Prepare model for k-bit training
        if self.quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,  # Use CAUSAL_LM for summarization
            inference_mode=False
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        return self.peft_model
    
    def get_model(self) -> nn.Module:
        """
        Get the configured model (loads if not already loaded).
        
        Returns:
            PEFT model
        """
        if self.peft_model is None:
            return self.load_model()
        return self.peft_model
    
    def get_tokenizer(self):
        """
        Get the tokenizer.
        
        Returns:
            Tokenizer instance
        """
        return self.tokenizer
    
    def save_adapter(self, output_dir: str):
        """
        Save only the LoRA adapter weights (not full model).
        
        Args:
            output_dir: Directory to save adapter weights
        """
        if self.peft_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Adapter saved to {output_dir}")
    
    def load_adapter(self, adapter_path: str):
        """
        Load LoRA adapter weights.
        
        Args:
            adapter_path: Path to saved adapter weights
        """
        if self.model is None:
            self.load_model()
        
        from peft import PeftModel
        self.peft_model = PeftModel.from_pretrained(
            self.model,
            adapter_path
        )
        print(f"Adapter loaded from {adapter_path}")
    
    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Extract trainable parameters (LoRA adapters only).
        
        Returns:
            Dictionary of parameter names to tensors
        """
        if self.peft_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        trainable_params = {}
        for name, param in self.peft_model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param.data.clone()
        
        return trainable_params
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set model parameters from dictionary.
        
        Args:
            parameters: Dictionary of parameter names to tensors
        """
        if self.peft_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        for name, param in self.peft_model.named_parameters():
            if name in parameters:
                param.data = parameters[name].clone()
    
    def generate_summary(
        self,
        text: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate summary for a legal document.
        
        Args:
            text: Input legal document text
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated summary
        """
        if self.peft_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.peft_model.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode summary
        summary = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return summary
