"""
BLEU Score Evaluation for Legal Document Summarization
Implements BLEU-1, BLEU-2, BLEU-3, BLEU-4 metrics.
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


def get_ngram_counts(tokens: List[str], n: int) -> Counter:
    """
    Get n-gram counts from tokens.
    
    Args:
        tokens: List of tokens
        n: N-gram order
        
    Returns:
        Counter of n-grams
    """
    ngrams = get_ngrams(tokens, n)
    return Counter(ngrams)


def compute_bleu_score(
    reference: List[str],
    candidate: List[str],
    n: int = 4,
    smoothing: bool = True
) -> float:
    """
    Compute BLEU score for a single reference-candidate pair.
    
    Args:
        reference: Reference token list
        candidate: Candidate token list
        n: Maximum n-gram order
        smoothing: Apply smoothing to avoid zero scores
        
    Returns:
        BLEU score (0-1)
    """
    if len(candidate) == 0:
        return 0.0
    
    if len(reference) == 0:
        return 0.0
    
    # Compute precision for each n-gram order
    precisions = []
    
    for i in range(1, n + 1):
        ref_ngrams = get_ngram_counts(reference, i)
        cand_ngrams = get_ngram_counts(candidate, i)
        
        if len(cand_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count matches
        matches = 0
        total = sum(cand_ngrams.values())
        
        for ngram, count in cand_ngrams.items():
            if ngram in ref_ngrams:
                matches += min(count, ref_ngrams[ngram])
        
        # Compute precision
        if smoothing and matches == 0:
            # Add-1 smoothing
            precision = 1.0 / (total + 1)
        else:
            precision = matches / total if total > 0 else 0.0
        
        precisions.append(precision)
    
    # Compute geometric mean of precisions
    if all(p > 0 for p in precisions):
        geometric_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        geometric_mean = 0.0
    
    # Compute brevity penalty
    if len(candidate) > len(reference):
        bp = 1.0
    else:
        bp = np.exp(1 - len(reference) / len(candidate)) if len(candidate) > 0 else 0.0
    
    # Final BLEU score
    bleu = bp * geometric_mean
    
    return bleu


def compute_bleu_1_to_4(
    references: List[List[str]],
    candidates: List[List[str]],
    smoothing: bool = True
) -> Dict[str, float]:
    """
    Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.
    
    Args:
        references: List of reference token lists
        candidates: List of candidate token lists
        smoothing: Apply smoothing
        
    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    if len(references) != len(candidates):
        raise ValueError("Number of references and candidates must match")
    
    bleu_scores = {f"bleu-{i}": [] for i in range(1, 5)}
    
    for ref, cand in zip(references, candidates):
        for n in range(1, 5):
            score = compute_bleu_score(ref, cand, n=n, smoothing=smoothing)
            bleu_scores[f"bleu-{n}"].append(score)
    
    # Average scores
    results = {k: np.mean(v) for k, v in bleu_scores.items()}
    
    return results


def evaluate_bleu(
    reference_texts: List[str],
    candidate_texts: List[str],
    tokenize: bool = True
) -> Dict[str, float]:
    """
    Evaluate BLEU scores for text summaries.
    
    Args:
        reference_texts: List of reference summary texts
        candidate_texts: List of candidate summary texts
        tokenize: Whether to tokenize texts (if False, assumes already tokenized)
        
    Returns:
        Dictionary with BLEU scores and statistics
    """
    # Tokenize if needed
    if tokenize:
        references = [word_tokenize(ref.lower()) for ref in reference_texts]
        candidates = [word_tokenize(cand.lower()) for cand in candidate_texts]
    else:
        references = reference_texts
        candidates = candidate_texts
    
    # Compute BLEU scores
    bleu_scores = compute_bleu_1_to_4(references, candidates)
    
    # Compute corpus-level BLEU (standard BLEU)
    corpus_bleu = compute_bleu_score(
        [token for ref in references for token in ref],
        [token for cand in candidates for token in cand],
        n=4
    )
    
    results = {
        **bleu_scores,
        "bleu-corpus": corpus_bleu,
        "num_samples": len(reference_texts)
    }
    
    return results


def evaluate_model_bleu(
    model,
    tokenizer,
    test_dataset,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate model BLEU scores on test dataset.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_dataset: Test dataset with 'text' and 'summary' fields
        device: Device to run evaluation on
        
    Returns:
        Dictionary with BLEU scores
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
    
    # Evaluate BLEU
    results = evaluate_bleu(references, candidates)
    
    return results
