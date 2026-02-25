"""
Client-side Metrics Computation for Federated Learning
Computes BLEU, ROUGE, and Perplexity metrics locally.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from evaluation.bleu import evaluate_bleu
from evaluation.rouge import evaluate_rouge


class ClientMetrics:
    """
    Computes evaluation metrics on client-side data.
    Ensures privacy by keeping all computations local.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize client metrics.

        Args:
            model: Trained model
            tokenizer: Model tokenizer
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def compute_perplexity(
        self,
        texts: List[str],
        batch_size: int = 4
    ) -> float:
        """
        Compute perplexity on given texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for computation

        Returns:
            Perplexity score
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)

                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss

                # Accumulate
                total_loss += loss.item() * len(batch_texts)
                total_tokens += inputs["input_ids"].numel()

        # Compute perplexity
        avg_loss = total_loss / len(texts)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity

    def generate_summaries(
        self,
        texts: List[str],
        max_length: int = 512,
        batch_size: int = 4
    ) -> List[str]:
        """
        Generate summaries for input texts.

        Args:
            texts: List of input texts
            max_length: Maximum summary length
            batch_size: Batch size for generation

        Returns:
            List of generated summaries
        """
        self.model.eval()
        summaries = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)

                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                # Decode
                batch_summaries = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                summaries.extend(batch_summaries)

        return summaries

    def evaluate_local_metrics(
        self,
        val_texts: List[str],
        val_summaries: List[str],
        batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Evaluate BLEU, ROUGE, and Perplexity on local validation data.

        Args:
            val_texts: Validation input texts
            val_summaries: Validation reference summaries
            batch_size: Batch size for computation

        Returns:
            Dictionary with all metrics
        """
        # Generate predictions
        pred_summaries = self.generate_summaries(val_texts, batch_size=batch_size)

        # Compute BLEU
        bleu_results = evaluate_bleu(val_summaries, pred_summaries)

        # Compute ROUGE
        rouge_results = evaluate_rouge(val_summaries, pred_summaries)

        # Compute Perplexity
        perplexity = self.compute_perplexity(val_texts, batch_size=batch_size)

        # Combine results
        metrics = {
            **bleu_results,
            **rouge_results,
            "perplexity": perplexity,
            "num_samples": len(val_texts)
        }

        return metrics

    def get_model_quality_score(
        self,
        metrics: Dict[str, float]
    ) -> float:
        """
        Compute overall model quality score from metrics.

        Args:
            metrics: Dictionary with BLEU, ROUGE, and perplexity

        Returns:
            Quality score (0-1, higher is better)
        """
        # Weights for different metrics
        weights = {
            "bleu-4": 0.3,
            "rouge-1-f1": 0.3,
            "rouge-l-f1": 0.3,
            "perplexity": 0.1
        }

        score = 0.0

        # BLEU and ROUGE (higher is better)
        if "bleu-4" in metrics:
            score += weights["bleu-4"] * min(metrics["bleu-4"], 1.0)

        if "rouge-1-f1" in metrics:
            score += weights["rouge-1-f1"] * metrics["rouge-1-f1"]

        if "rouge-l-f1" in metrics:
            score += weights["rouge-l-f1"] * metrics["rouge-l-f1"]

        # Perplexity (lower is better, normalize to 0-1)
        if "perplexity" in metrics:
            # Normalize perplexity (assuming 1-100 range)
            norm_perplexity = max(0, min(1, (100 - metrics["perplexity"]) / 99))
            score += weights["perplexity"] * norm_perplexity

        return score


def compute_client_metrics(
    model: nn.Module,
    tokenizer,
    val_dataset,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Convenience function to compute client metrics.

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        val_dataset: Validation dataset with 'text' and 'summary' fields
        device: Device for computation

    Returns:
        Dictionary with metrics
    """
    metrics_computer = ClientMetrics(model, tokenizer, device)

    # Extract texts and summaries
    val_texts = [ex["text"] for ex in val_dataset]
    val_summaries = [ex["summary"] for ex in val_dataset]

    # Compute metrics
    results = metrics_computer.evaluate_local_metrics(val_texts, val_summaries)

    return results