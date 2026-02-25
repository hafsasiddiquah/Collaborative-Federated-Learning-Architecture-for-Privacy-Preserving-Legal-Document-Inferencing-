"""
Model Registry and Version Control for Federated Learning
Manages model versions, metrics tracking, and deployment decisions.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


class ModelRegistry:
    """
    Manages model versions, metrics, and deployment decisions.
    Provides versioning, rollback, and automated deployment capabilities.
    """

    def __init__(
        self,
        registry_path: str = "./models",
        max_versions: int = 10,
        auto_cleanup: bool = True
    ):
        """
        Initialize model registry.

        Args:
            registry_path: Path to store model versions
            max_versions: Maximum number of versions to keep
            auto_cleanup: Whether to automatically clean up old versions
        """
        self.registry_path = Path(registry_path)
        self.max_versions = max_versions
        self.auto_cleanup = auto_cleanup

        # Create registry directory
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Registry metadata
        self.metadata_file = self.registry_path / "registry.json"
        self.metrics_file = self.registry_path / "metrics.csv"

        # Load or initialize metadata
        self.metadata = self._load_metadata()
        self.metrics_df = self._load_metrics()

    def _load_metadata(self) -> Dict:
        """Load registry metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "versions": {},
                "current_version": None,
                "total_versions": 0
            }

    def _save_metadata(self):
        """Save registry metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _load_metrics(self) -> pd.DataFrame:
        """Load metrics dataframe."""
        if self.metrics_file.exists():
            return pd.read_csv(self.metrics_file)
        else:
            return pd.DataFrame(columns=[
                "version", "timestamp", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
                "rouge_1_precision", "rouge_1_recall", "rouge_1_f1",
                "rouge_2_precision", "rouge_2_recall", "rouge_2_f1",
                "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
                "perplexity", "epsilon", "deployed"
            ])

    def _save_metrics(self):
        """Save metrics dataframe."""
        self.metrics_df.to_csv(self.metrics_file, index=False)

    def register_model(
        self,
        model_path: str,
        metrics: Dict[str, float],
        epsilon: Optional[float] = None,
        round_number: Optional[int] = None
    ) -> str:
        """
        Register a new model version.

        Args:
            model_path: Path to the model files
            metrics: Dictionary with evaluation metrics
            epsilon: Privacy budget spent
            round_number: Federated learning round number

        Returns:
            Version identifier
        """
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{self.metadata['total_versions'] + 1}_{timestamp}"

        # Create version directory
        version_path = self.registry_path / version
        version_path.mkdir(parents=True, exist_ok=True)

        # Copy model files
        if os.path.isdir(model_path):
            for file in os.listdir(model_path):
                src = os.path.join(model_path, file)
                dst = version_path / file
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        else:
            # Single file
            shutil.copy2(model_path, version_path / "model.bin")

        # Update metadata
        self.metadata["versions"][version] = {
            "path": str(version_path),
            "timestamp": datetime.now().isoformat(),
            "round": round_number,
            "epsilon": epsilon,
            "metrics": metrics
        }

        self.metadata["total_versions"] += 1
        self._save_metadata()

        # Add to metrics dataframe
        metrics_row = {
            "version": version,
            "timestamp": datetime.now(),
            "bleu_1": metrics.get("bleu-1", 0.0),
            "bleu_2": metrics.get("bleu-2", 0.0),
            "bleu_3": metrics.get("bleu-3", 0.0),
            "bleu_4": metrics.get("bleu-4", 0.0),
            "rouge_1_precision": metrics.get("rouge-1-precision", 0.0),
            "rouge_1_recall": metrics.get("rouge-1-recall", 0.0),
            "rouge_1_f1": metrics.get("rouge-1-f1", 0.0),
            "rouge_2_precision": metrics.get("rouge-2-precision", 0.0),
            "rouge_2_recall": metrics.get("rouge-2-recall", 0.0),
            "rouge_2_f1": metrics.get("rouge-2-f1", 0.0),
            "rouge_l_precision": metrics.get("rouge-l-precision", 0.0),
            "rouge_l_recall": metrics.get("rouge-l-recall", 0.0),
            "rouge_l_f1": metrics.get("rouge-l-f1", 0.0),
            "perplexity": metrics.get("perplexity", float('inf')),
            "epsilon": epsilon,
            "deployed": False
        }

        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([metrics_row])],
                                   ignore_index=True)
        self._save_metrics()

        # Auto cleanup if enabled
        if self.auto_cleanup:
            self._cleanup_old_versions()

        return version

    def should_deploy(
        self,
        metrics: Dict[str, float],
        bleu_threshold: float = 0.3,
        rouge_threshold: float = 0.4,
        perplexity_threshold: float = 50.0
    ) -> bool:
        """
        Determine if model should be deployed based on metrics.

        Args:
            metrics: Model evaluation metrics
            bleu_threshold: Minimum BLEU-4 score
            rouge_threshold: Minimum ROUGE-1 F1 score
            perplexity_threshold: Maximum perplexity

        Returns:
            True if model should be deployed
        """
        bleu_score = metrics.get("bleu-4", 0.0)
        rouge_score = metrics.get("rouge-1-f1", 0.0)
        perplexity = metrics.get("perplexity", float('inf'))

        return (
            bleu_score >= bleu_threshold and
            rouge_score >= rouge_threshold and
            perplexity <= perplexity_threshold
        )

    def deploy_model(self, version: str) -> bool:
        """
        Mark a model version as deployed.

        Args:
            version: Version to deploy

        Returns:
            True if deployment successful
        """
        if version not in self.metadata["versions"]:
            return False

        # Update current version
        self.metadata["current_version"] = version

        # Mark as deployed in metrics
        self.metrics_df.loc[self.metrics_df["version"] == version, "deployed"] = True

        self._save_metadata()
        self._save_metrics()

        return True

    def get_current_model(self) -> Optional[Dict]:
        """
        Get information about the currently deployed model.

        Returns:
            Model information dictionary or None
        """
        current_version = self.metadata.get("current_version")
        if current_version and current_version in self.metadata["versions"]:
            return self.metadata["versions"][current_version]
        return None

    def get_model_info(self, version: str) -> Optional[Dict]:
        """
        Get information about a specific model version.

        Args:
            version: Version identifier

        Returns:
            Model information dictionary or None
        """
        return self.metadata["versions"].get(version)

    def list_versions(self) -> List[Dict]:
        """
        List all model versions with their information.

        Returns:
            List of version information dictionaries
        """
        versions = []
        for version, info in self.metadata["versions"].items():
            version_info = info.copy()
            version_info["version"] = version
            versions.append(version_info)

        # Sort by timestamp (newest first)
        versions.sort(key=lambda x: x["timestamp"], reverse=True)

        return versions

    def rollback_to_version(self, version: str) -> bool:
        """
        Rollback to a previous model version.

        Args:
            version: Version to rollback to

        Returns:
            True if rollback successful
        """
        if version not in self.metadata["versions"]:
            return False

        # Update current version
        self.metadata["current_version"] = version

        # Mark new version as deployed
        self.metrics_df.loc[self.metrics_df["version"] == version, "deployed"] = True

        self._save_metadata()
        self._save_metrics()

        return True

    def get_best_model(
        self,
        metric: str = "bleu_4",
        minimize: bool = False
    ) -> Optional[str]:
        """
        Get the best model version based on a metric.

        Args:
            metric: Metric to optimize
            minimize: Whether to minimize (True) or maximize (False) the metric

        Returns:
            Best version identifier or None
        """
        if self.metrics_df.empty:
            return None

        if metric not in self.metrics_df.columns:
            return None

        if minimize:
            best_idx = self.metrics_df[metric].idxmin()
        else:
            best_idx = self.metrics_df[metric].idxmax()

        return self.metrics_df.loc[best_idx, "version"]

    def get_metrics_history(self) -> pd.DataFrame:
        """
        Get the complete metrics history.

        Returns:
            DataFrame with metrics history
        """
        return self.metrics_df.copy()

    def get_performance_trend(
        self,
        metric: str = "bleu_4",
        window: int = 5
    ) -> Dict[str, float]:
        """
        Analyze performance trend for a metric.

        Args:
            metric: Metric to analyze
            window: Number of recent versions to consider

        Returns:
            Dictionary with trend analysis
        """
        if self.metrics_df.empty or metric not in self.metrics_df.columns:
            return {"trend": "insufficient_data", "slope": 0.0, "improving": False}

        # Get recent metrics
        recent = self.metrics_df.tail(window)[metric].values

        if len(recent) < 2:
            return {"trend": "insufficient_data", "slope": 0.0, "improving": False}

        # Calculate trend
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        # Determine if improving (for BLEU/ROUGE, positive slope is good)
        improving = slope > 0

        trend = "improving" if improving else "declining"

        return {
            "trend": trend,
            "slope": float(slope),
            "improving": improving,
            "recent_avg": float(np.mean(recent)),
            "recent_std": float(np.std(recent))
        }

    def _cleanup_old_versions(self):
        """Clean up old model versions to save space."""
        versions = self.list_versions()

        if len(versions) <= self.max_versions:
            return

        # Keep the most recent versions and any deployed versions
        deployed_versions = set(
            self.metrics_df[self.metrics_df["deployed"]]["version"].tolist()
        )

        versions_to_keep = set()
        for version_info in versions[:self.max_versions]:
            versions_to_keep.add(version_info["version"])

        # Always keep deployed versions
        versions_to_keep.update(deployed_versions)

        # Remove old versions
        for version_info in versions[self.max_versions:]:
            version = version_info["version"]
            if version not in versions_to_keep:
                version_path = Path(version_info["path"])
                if version_path.exists():
                    shutil.rmtree(version_path)

                # Remove from metadata
                if version in self.metadata["versions"]:
                    del self.metadata["versions"][version]

        self._save_metadata()

    def export_registry(self, export_path: str):
        """
        Export registry data for backup or analysis.

        Args:
            export_path: Path to export registry data
        """
        export_data = {
            "metadata": self.metadata,
            "metrics": self.metrics_df.to_dict('records')
        }

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

    def import_registry(self, import_path: str):
        """
        Import registry data from backup.

        Args:
            import_path: Path to import registry data from
        """
        with open(import_path, 'r') as f:
            import_data = json.load(f)

        self.metadata = import_data["metadata"]
        self.metrics_df = pd.DataFrame(import_data["metrics"])

        self._save_metadata()
        self._save_metrics()