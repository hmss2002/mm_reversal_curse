"""Evaluation metrics."""
from dataclasses import dataclass, field
from typing import Dict, List, Any
import json
from pathlib import Path


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics."""
    
    forward_accuracy: float = 0.0
    forward_samples: int = 0
    reverse_accuracy: float = 0.0
    reverse_samples: int = 0
    random_baseline: float = 0.25  # 1/4 for 4-way MCQA
    
    forward_predictions: List[Dict] = field(default_factory=list)
    reverse_predictions: List[Dict] = field(default_factory=list)
    
    def compute_reversal_gap(self) -> float:
        """Compute the gap between forward and reverse accuracy."""
        return self.forward_accuracy - self.reverse_accuracy
    
    def is_reversal_curse_detected(self) -> bool:
        """Check if reverse accuracy is close to random baseline."""
        return self.reverse_accuracy <= self.random_baseline + 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "forward_accuracy": self.forward_accuracy,
            "forward_samples": self.forward_samples,
            "reverse_accuracy": self.reverse_accuracy,
            "reverse_samples": self.reverse_samples,
            "random_baseline": self.random_baseline,
            "reversal_gap": self.compute_reversal_gap(),
            "reversal_curse_detected": self.is_reversal_curse_detected()
        }
    
    def save(self, path: str):
        """Save metrics to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        """Return formatted summary."""
        return f"""
=== Experiment Results ===
Forward Accuracy: {self.forward_accuracy:.2%} ({self.forward_samples} samples)
Reverse Accuracy: {self.reverse_accuracy:.2%} ({self.reverse_samples} samples)
Random Baseline:  {self.random_baseline:.2%}
Reversal Gap:     {self.compute_reversal_gap():.2%}
Reversal Curse:   {'DETECTED' if self.is_reversal_curse_detected() else 'NOT DETECTED'}
"""
