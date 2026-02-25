"""
ROUGE Score Evaluation for Legal Document Summarization
Implements ROUGE-1, ROUGE-2, ROUGE-L metrics.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import warnings

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Extract n-grams from token list.
    
    Args:
        tokens: List of tokens
        n: N-gram order
        
    Returns:
        List of n-gram tuples
    """
    if len(tokens) < n:
        return []
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute length of longest common subsequence (LCS).
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        LCS length
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def compute_rouge_n(
    reference: List[str],
    candidate: List[str],
    n: int = 1
) -> Dict[str, float]:
    """
    Compute ROUGE-N precision, recall, and F1.
    
    Args:
        reference: Reference token list
        candidate: Candidate token list
        n: N-gram order
        
    Returns:
        Dictionary with precision, recall, and f1
    """
    if len(candidate) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if len(reference) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Get n-grams
    ref_ngrams = Counter(get_ngrams(reference, n))
    cand_ngrams = Counter(get_ngrams(candidate, n))
    
    if len(cand_ngrams) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Count matches
    matches = 0
    for ngram, count in cand_ngrams.items():
        if ngram in ref_ngrams:
            matches += min(count, ref_ngrams[ngram])
    
    # Compute precision and recall
    precision = matches / len(cand_ngrams) if len(cand_ngrams) > 0 else 0.0
    recall = matches / len(ref_ngrams) if len(ref_ngrams) > 0 else 0.0
    
    # Compute F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_rouge_l(
    reference: List[str],
    candidate: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE-L precision, recall, and F1.
    
    Args:
        reference: Reference token list
        candidate: Candidate token list
        
    Returns:
        Dictionary with precision, recall, and f1
    """
    if len(candidate) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if len(reference) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Compute LCS
    lcs_length = longest_common_subsequence(reference, candidate)
    
    # Compute precision and recall
    precision = lcs_length / len(candidate) if len(candidate) > 0 else 0.0
    recall = lcs_length / len(reference) if len(reference) > 0 else 0.0
    
    # Compute F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_rouge_scores(
    references: List[List[str]],
    candidates: List[List[str]]
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.
    
    Args:
        references: List of reference token lists
        candidates: List of candidate token lists
        
    Returns:
        Dictionary with all ROUGE metrics
    """
    if len(references) != len(candidates):
        raise ValueError("Number of references and candidates must match")
    
    rouge_1_scores = {"precision": [], "recall": [], "f1": []}
    rouge_2_scores = {"precision": [], "recall": [], "f1": []}
    rouge_l_scores = {"precision": [], "recall": [], "f1": []}
    
    for ref, cand in zip(references, candidates):
        # ROUGE-1
        r1 = compute_rouge_n(ref, cand, n=1)
        for key in ["precision", "recall", "f1"]:
            rouge_1_scores[key].append(r1[key])
        
        # ROUGE-2
        r2 = compute_rouge_n(ref, cand, n=2)
        for key in ["precision", "recall", "f1"]:
            rouge_2_scores[key].append(r2[key])
        
        # ROUGE-L
        rl = compute_rouge_l(ref, cand)
        for key in ["precision", "recall", "f1"]:
            rouge_l_scores[key].append(rl[key])
    
    # Average scores
    results = {}
    
    for metric in ["precision", "recall", "f1"]:
        results[f"rouge-1-{metric}"] = np.mean(rouge_1_scores[metric])
        results[f"rouge-2-{metric}"] = np.mean(rouge_2_scores[metric])
        results[f"rouge-l-{metric}"] = np.mean(rouge_l_scores[metric])
    
    return results


def evaluate_rouge(
    reference_texts: List[str],
    candidate_texts: List[str],
    tokenize: bool = True
) -> Dict[str, float]:
    """
    Evaluate ROUGE scores for text summaries.
    
    Args:
        reference_texts: List of reference summary texts
        candidate_texts: List of candidate summary texts
        tokenize: Whether to tokenize texts (if False, assumes already tokenized)
        
    Returns:
        Dictionary with ROUGE scores and statistics
    """
    # Tokenize if needed
    if tokenize:
        references = [word_tokenize(ref.lower()) for ref in reference_texts]
        candidates = [word_tokenize(cand.lower()) for cand in candidate_texts]
    else:
        references = reference_texts
        candidates = candidate_texts
    
    # Compute ROUGE scores
    rouge_scores = compute_rouge_scores(references, candidates)
    
    results = {
        **rouge_scores,
        "num_samples": len(reference_texts)
    }
    
    return results


def evaluate_model_rouge(
    model,
    tokenizer,
    test_dataset,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate model ROUGE scores on test dataset.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_dataset: Test dataset with 'text' and 'summary' fields
        device: Device to run evaluation on
        
    Returns:
        Dictionary with ROUGE scores
    """
    model.eval()
    references = []
    candidates = []
    
    import torch
    
    with torch.no_grad():
        for example in test_dataset:
            # Get reference
            ref_text = example.get("summary", "")
            references.append(ref_text)
            
            # Generate candidate
            input_text = example.get("text", "")
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)
            
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            
            cand_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            candidates.append(cand_text)
    
    # Evaluate ROUGE
    results = evaluate_rouge(references, candidates)
    
    return results


class ROUGEScore:
    """
    ROUGE Score evaluator class.
    """
    
    def __init__(self):
        pass
    
    def score(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            references: List of reference summaries
            candidates: List of candidate summaries
            
        Returns:
            Dictionary with ROUGE scores
        """
        return evaluate_rouge(references, candidates)
