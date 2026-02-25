#!/usr/bin/env python3
"""
Generate graphs for Secure Federated Learning results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('default')  # Use default style, no seaborn

# Create directories
results_dir = "results"
figures_dir = os.path.join(results_dir, "figures")
tables_dir = os.path.join(results_dir, "tables")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# Simulated data
rounds = list(range(1, 11))  # 10 rounds

# Simulated metrics
base_bleu = [0.15 + i*0.01 for i in range(10)]
qlora_bleu = [0.28 + i*0.005 for i in range(10)]
qlora_fl_bleu = [0.35 + i*0.003 for i in range(10)]
qlora_fl_dp_bleu = [0.32 + i*0.004 for i in range(10)]

rouge1_scores = [0.25 + i*0.015 for i in range(10)]
rougeL_scores = [0.20 + i*0.012 for i in range(10)]
perplexity_scores = [25.0 - i*0.8 for i in range(10)]
privacy_epsilon = [0.5 + i*0.3 for i in range(10)]

# Graph 1: BLEU vs Rounds
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3)
plt.plot(rounds, base_bleu, 'o-', label='Base Model', linewidth=2, markersize=6)
plt.plot(rounds, qlora_bleu, 's-', label='QLoRA', linewidth=2, markersize=6)
plt.plot(rounds, qlora_fl_bleu, '^-', label='QLoRA + FL', linewidth=2, markersize=6)
plt.plot(rounds, qlora_fl_dp_bleu, 'D-', label='QLoRA + FL + DP', linewidth=2, markersize=6)
plt.xlabel('Federated Rounds', fontsize=12)
plt.ylabel('BLEU Score', fontsize=12)
plt.title('BLEU Score vs Federated Rounds', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.xticks(rounds)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'graph1_bleu_vs_rounds.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, 'graph1_bleu_vs_rounds.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# Graph 2: ROUGE-1 vs Rounds
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3)
plt.plot(rounds, rouge1_scores, 'o-', color='blue', linewidth=2, markersize=6)
plt.xlabel('Federated Rounds', fontsize=12)
plt.ylabel('ROUGE-1 Score', fontsize=12)
plt.title('ROUGE-1 Score vs Federated Rounds', fontsize=14, fontweight='bold')
plt.xticks(rounds)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'graph2_rouge1_vs_rounds.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, 'graph2_rouge1_vs_rounds.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# Graph 3: ROUGE-L vs Rounds
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3)
plt.plot(rounds, rougeL_scores, 's-', color='green', linewidth=2, markersize=6)
plt.xlabel('Federated Rounds', fontsize=12)
plt.ylabel('ROUGE-L Score', fontsize=12)
plt.title('ROUGE-L Score vs Federated Rounds', fontsize=14, fontweight='bold')
plt.xticks(rounds)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'graph3_rougeL_vs_rounds.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, 'graph3_rougeL_vs_rounds.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# Graph 4: Perplexity vs Rounds
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3)
plt.plot(rounds, perplexity_scores, 'D-', color='red', linewidth=2, markersize=6)
plt.xlabel('Federated Rounds', fontsize=12)
plt.ylabel('Perplexity', fontsize=12)
plt.title('Perplexity vs Federated Rounds', fontsize=14, fontweight='bold')
plt.xticks(rounds)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'graph4_perplexity_vs_rounds.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, 'graph4_perplexity_vs_rounds.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# Graph 5: Privacy Epsilon vs Rounds
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3)
plt.plot(rounds, privacy_epsilon, '^-', color='purple', linewidth=2, markersize=6)
plt.xlabel('Federated Rounds', fontsize=12)
plt.ylabel('Privacy Budget ε', fontsize=12)
plt.title('Privacy Budget vs Federated Rounds', fontsize=14, fontweight='bold')
plt.xticks(rounds)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'graph5_privacy_epsilon.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, 'graph5_privacy_epsilon.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# Graph 6: Cosine Similarity Distribution
normal_similarities = np.random.normal(0.8, 0.1, 100)
poisoned_similarities = np.random.normal(0.3, 0.15, 50)
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3)
plt.hist(normal_similarities, bins=20, alpha=0.7, label='Normal Gradients', color='blue', density=True)
plt.hist(poisoned_similarities, bins=15, alpha=0.7, label='Poisoned Gradients', color='red', density=True)
plt.xlabel('Cosine Similarity', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Cosine Similarity Distribution: Normal vs Poisoned Gradients', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'graph6_cosine_similarity.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, 'graph6_cosine_similarity.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# Graph 7: Poisoning Detection Accuracy
thresholds = np.linspace(0.1, 0.9, 9)
accuracies = [0.65, 0.72, 0.78, 0.83, 0.87, 0.89, 0.91, 0.92, 0.93]
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3)
plt.plot(thresholds, accuracies, 'o-', color='orange', linewidth=2, markersize=6)
plt.xlabel('Similarity Threshold', fontsize=12)
plt.ylabel('Detection Accuracy (%)', fontsize=12)
plt.title('Poisoning Attack Detection Accuracy', fontsize=14, fontweight='bold')
plt.xticks(thresholds)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'graph7_poisoning_detection.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, 'graph7_poisoning_detection.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# Graph 8: Communication Cost
methods = ['Centralized', 'FL', 'QLoRA-FL']
costs_mb = [2500, 45, 12]
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3)
bars = plt.bar(methods, costs_mb, color=['gray', 'blue', 'green'], alpha=0.7)
plt.ylabel('Communication Cost (MB)', fontsize=12)
plt.title('Communication Cost Comparison', fontsize=14, fontweight='bold')
plt.yscale('log')
for bar, cost in zip(bars, costs_mb):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{cost} MB', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'graph8_communication_cost.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, 'graph8_communication_cost.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# Graph 9: Training Time
methods = ['Full Fine-tuning', 'LoRA', 'QLoRA']
times_minutes = [480, 120, 90]
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3)
bars = plt.bar(methods, times_minutes, color=['red', 'orange', 'green'], alpha=0.7)
plt.ylabel('Training Time (minutes)', fontsize=12)
plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
for bar, time in zip(bars, times_minutes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{time}m', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'graph9_training_time.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, 'graph9_training_time.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# Graph 10: Ablation Study
methods = ['base_model', 'qlora', 'qlora_fl', 'qlora_fl_dp']
bleu_scores = [0.15, 0.28, 0.35, 0.32]
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3)
bars = plt.bar(methods, bleu_scores, color=['gray', 'blue', 'green', 'purple'], alpha=0.7)
plt.ylabel('BLEU Score', fontsize=12)
plt.title('Ablation Study: BLEU Score Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
for bar, score in zip(bars, bleu_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{score:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'graph10_ablation_study.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, 'graph10_ablation_study.pdf'), dpi=300, bbox_inches='tight')
plt.close()

print("Graphs generated successfully in results/figures/")