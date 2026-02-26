"""
Legal Dataset Loader for Federated Learning
Loads and preprocesses legal documents from HuggingFace pile-of-law dataset.
"""

import os
import json
import warnings
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress HuggingFace warnings
warnings.filterwarnings('ignore', category=UserWarning)


class LegalDatasetLoader:
    """
    Loads and preprocesses legal documents for summarization tasks.
    Supports pile-of-law dataset from HuggingFace.
    """
    
    def __init__(
        self,
        dataset_name: str = "pile-of-law/pile-of-law",
        tokenizer_name: str = "google/pegasus-xsum",
        max_length: int = 2048,
        max_target_length: int = 512,
        cache_dir: str = "./data_cache"
    ):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            tokenizer_name: Tokenizer model name
            max_length: Maximum input sequence length
            max_target_length: Maximum target summary length
            cache_dir: Directory to cache datasets
        """
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.cache_dir = cache_dir
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def load_dataset(
        self,
        dataset_name: str = None,
        split: str = "train",
        num_samples: int = None,
        config: str = None
    ) -> Dataset:
        """
        Load dataset with auto-detection of available splits and configs.
        Handles pile-of-law which requires a specific config name.

        Args:
            dataset_name: HuggingFace dataset name (uses self.dataset_name if None)
            split: Preferred split to load (auto-detects if not available)
            num_samples: Limit number of samples
            config: Dataset configuration (e.g., 'all', 'courtlistener_opinions' for pile-of-law)

        Returns:
            Loaded and validated dataset
        """
        dataset_name = dataset_name or self.dataset_name

        # Special handling for pile-of-law dataset - requires config name
        if "pile-of-law" in dataset_name.lower():
            config = config or "courtlistener_opinions"  # Default to a reasonable subset
            logger.info(f"Using config '{config}' for pile-of-law dataset")

        max_retries = 3
        dataset = None

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Loading {dataset_name}" +
                           (f" with config '{config}'" if config else ""))

                # Try to load with specified split and config
                try:
                    load_kwargs = {
                        "cache_dir": self.cache_dir,
                        "trust_remote_code": True,
                        "num_proc": 1
                    }

                    if config:
                        load_kwargs["name"] = config

                    dataset = load_dataset(
                        dataset_name,
                        split=split,
                        **load_kwargs
                    )
                    logger.info(f"✓ Successfully loaded split '{split}'" +
                               (f" with config '{config}'" if config else ""))
                    break
                except Exception as e:
                    logger.warning(f"Failed to load split '{split}': {str(e)[:100]}...")

                    # Try to load all splits and select the first/largest one
                    logger.info("Attempting to load all available splits...")
                    load_kwargs = {
                        "cache_dir": self.cache_dir,
                        "trust_remote_code": True,
                        "num_proc": 1
                    }

                    if config:
                        load_kwargs["name"] = config

                    dataset_dict = load_dataset(
                        dataset_name,
                        **load_kwargs
                    )

                    if isinstance(dataset_dict, dict):
                        # Get the largest split
                        split_name = max(dataset_dict.keys(), key=lambda x: len(dataset_dict[x]))
                        dataset = dataset_dict[split_name]
                        logger.info(f"✓ Loaded split '{split_name}' with {len(dataset)} samples")
                        break
                    else:
                        dataset = dataset_dict
                        break

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)[:100]}...")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to load dataset after {max_retries} attempts")

        # Fallback to sample data
        if dataset is None:
            logger.warning("⚠️  Falling back to sample data generation...")
            dataset = self._generate_sample_data(num_samples or 100)
        else:
            # Validate and fix dataset structure
            dataset = self._validate_and_fix_dataset(dataset)

            # Limit samples if specified
            if num_samples and len(dataset) > num_samples:
                logger.info(f"Limiting dataset to {num_samples} samples")
                dataset = dataset.select(range(num_samples))

        return dataset

    def load_local_dataset(
        self,
        data_path: str,
        split: str = "train",
        num_samples: int = None
    ) -> Dataset:
        """
        Load legal documents from local path or HuggingFace.
        
        Args:
            data_path: Path to local dataset or HuggingFace dataset name
            split: Dataset split to load
            num_samples: Limit number of samples (for testing)
            
        Returns:
            Preprocessed dataset
        """
        dataset = None

        # First try: Load from local directory
        if os.path.exists(data_path) and os.path.isdir(data_path):
            try:
                logger.info(f"Loading dataset from local path: {data_path}")
                dataset = load_dataset(
                    "json",
                    data_files=os.path.join(data_path, "*.json"),
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                if isinstance(dataset, dict):
                    dataset = dataset.get(split, dataset.get('train', list(dataset.values())[0]))
                logger.info(f"✓ Successfully loaded {len(dataset)} samples from local path")
            except Exception as e:
                logger.warning(f"Failed to load from local path: {e}")
                dataset = None

        # Second try: Load from HuggingFace with retry logic
        if dataset is None:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Loading dataset from HuggingFace: {data_path} (Attempt {attempt + 1}/{max_retries})")
                    dataset = load_dataset(
                        data_path,
                        split=split,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                        num_proc=1  # Single process for stability
                    )
                    logger.info(f"✓ Successfully loaded {len(dataset)} samples from HuggingFace")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to load from HuggingFace after {max_retries} attempts")
                        dataset = None

        # Third fallback: Generate sample data
        if dataset is None:
            logger.warning("⚠️  Falling back to sample data generation...")
            dataset = self._generate_sample_data(num_samples or 100)

        # Validate dataset structure and fix column names if needed
        dataset = self._validate_and_fix_dataset(dataset)

        # Limit number of samples if specified
        if num_samples and len(dataset) > num_samples:
            logger.info(f"Limiting dataset to {num_samples} samples")
            dataset = dataset.select(range(num_samples))

        return dataset
    
    def _generate_sample_data(self, num_samples: int) -> Dataset:
        """
        Generate sample legal document data for testing.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dataset with sample legal documents
        """
        sample_data = []
        
        legal_templates = [
            {
                "text": "IN THE SUPREME COURT OF THE UNITED STATES\n\nCase No. 2024-001\n\nPetitioner v. Respondent\n\nBRIEF FOR PETITIONER\n\nThe petitioner respectfully submits this brief in support of their motion for summary judgment. The central issue before this Court concerns the interpretation of contractual obligations under state law. The petitioner argues that the respondent failed to fulfill their contractual duties as specified in Section 4.2 of the agreement dated January 15, 2023. The agreement explicitly requires the respondent to provide quarterly financial reports within thirty days of each quarter's end. The respondent's failure to comply constitutes a material breach of contract, entitling the petitioner to seek damages and termination of the agreement.",
                "summary": "Petitioner seeks summary judgment for respondent's breach of contract by failing to provide required quarterly financial reports."
            },
            {
                "text": "CONFIDENTIAL SETTLEMENT AGREEMENT\n\nThis Settlement Agreement (\"Agreement\") is entered into on [DATE] between Party A and Party B. WHEREAS, the parties wish to resolve all disputes arising from the contract dated [DATE]; NOW THEREFORE, the parties agree as follows: 1. Party B shall pay Party A the sum of $500,000 within 30 days. 2. Both parties release all claims against each other. 3. This agreement is confidential and non-disclosure applies. 4. Governing law is the State of New York.",
                "summary": "Settlement agreement: Party B pays $500,000 to Party A, mutual release of claims, confidential, NY law applies."
            }
        ]
        
        for i in range(num_samples):
            template = legal_templates[i % len(legal_templates)]
            sample_data.append({
                "id": f"legal_doc_{i}",
                "text": template["text"],
                "summary": template["summary"],
                "document_type": ["case", "contract", "petition", "judgment"][i % 4]
            })
        
        return Dataset.from_list(sample_data)
    
    def _validate_and_fix_dataset(self, dataset: Dataset) -> Dataset:
        """
        Validate dataset structure and fix/normalize column names.

        Args:
            dataset: Input dataset to validate

        Returns:
            Dataset with normalized column names
        """
        if not isinstance(dataset, Dataset):
            logger.error("Invalid dataset type")
            return self._generate_sample_data(100)

        if len(dataset) == 0:
            logger.warning("Empty dataset, generating sample data")
            return self._generate_sample_data(100)

        # Check and normalize column names
        columns = dataset.column_names
        logger.info(f"Dataset columns: {columns}")

        # Map common column name variations to standard names
        column_mapping = {}

        # Look for text/document column
        text_candidates = ['text', 'document', 'content', 'body', 'passage', 'doc']
        summary_candidates = ['summary', 'target', 'label', 'headline', 'title']

        text_col = None
        for candidate in text_candidates:
            if candidate in columns:
                text_col = candidate
                break

        summary_col = None
        for candidate in summary_candidates:
            if candidate in columns:
                summary_col = candidate
                break

        # If we can't find expected columns, try to auto-detect
        if text_col is None and len(columns) > 0:
            logger.warning(f"No standard text column found. Available columns: {columns}")
            # Use first non-metadata column as text
            text_col = [c for c in columns if c not in ['id', 'metadata', 'type', 'source']][0] if columns else 'text'

        if summary_col is None and len(columns) > 1:
            # Use second column as summary if available
            summary_col = [c for c in columns if c != text_col and c not in ['id', 'metadata', 'type', 'source']]
            summary_col = summary_col[0] if summary_col else 'summary'

        # Rename columns if necessary
        if text_col and text_col != 'text':
            logger.info(f"Renaming '{text_col}' to 'text'")
            dataset = dataset.rename_column(text_col, 'text')

        if summary_col and summary_col != 'summary':
            logger.info(f"Renaming '{summary_col}' to 'summary'")
            dataset = dataset.rename_column(summary_col, 'summary')

        # Ensure both text and summary columns exist
        if 'text' not in dataset.column_names:
            logger.warning("'text' column not found, adding placeholder")
            dataset = dataset.add_column('text', ['Legal document text'] * len(dataset))

        if 'summary' not in dataset.column_names:
            logger.warning("'summary' column not found, adding placeholder")
            dataset = dataset.add_column('summary', ['Document summary'] * len(dataset))

        # Validate data types - ensure text and summary are strings
        def ensure_strings(example):
            example['text'] = str(example.get('text', ''))
            example['summary'] = str(example.get('summary', ''))
            return example

        dataset = dataset.map(ensure_strings, desc="Validating data types")

        # Remove samples with empty text or summary
        def has_content(example):
            return len(example['text']) > 0 and len(example['summary']) > 0

        original_len = len(dataset)
        dataset = dataset.filter(has_content, desc="Filtering empty samples")
        removed = original_len - len(dataset)
        if removed > 0:
            logger.info(f"Removed {removed} samples with empty text or summary")

        logger.info(f"✓ Dataset validation complete. Final size: {len(dataset)}")
        return dataset

    def preprocess_function(
        self,
        examples: Dict[str, List]
    ) -> Dict[str, List]:
        """
        Tokenize and preprocess legal documents for summarization.
        
        Args:
            examples: Batch of examples with 'text' and 'summary' fields
            
        Returns:
            Tokenized and formatted examples
        """
        # Extract text and summaries with fallbacks
        if "text" in examples:
            texts = examples["text"]
        elif "document" in examples:
            texts = examples["document"]
        else:
            texts = list(examples.values())[0]

        if "summary" in examples:
            summaries = examples["summary"]
        elif "target" in examples:
            summaries = examples["target"]
        else:
            summaries = ["Legal document summary"] * len(texts)
        
        # Ensure lists are not empty
        if not texts:
            texts = [""] * len(summaries) if summaries else [""]
        if not summaries:
            summaries = [""] * len(texts)

        # Convert to strings and handle None values
        texts = [str(t) if t is not None else "" for t in texts]
        summaries = [str(s) if s is not None else "" for s in summaries]

        # Tokenize inputs
        try:
            model_inputs = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Error tokenizing inputs: {e}")
            raise

        # Tokenize targets
        try:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    summaries,
                    max_length=self.max_target_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
        except Exception as e:
            logger.error(f"Error tokenizing targets: {e}")
            raise

        # Replace padding token id with -100 for loss calculation
        labels = labels["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create PyTorch DataLoader from preprocessed dataset.
        
        Args:
            dataset: Preprocessed dataset
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            
        Returns:
            PyTorch DataLoader
        """
        # Preprocess dataset
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Set format for PyTorch
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        return dataloader
    
    def split_dataset(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: Full dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            Tuple of (train, val, test) datasets
        """
        dataset = dataset.shuffle(seed=42)
        total_size = len(dataset)
        
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))
        
        return train_dataset, val_dataset, test_dataset
