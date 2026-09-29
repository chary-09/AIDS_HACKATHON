"""Metrics and confusion-matrix utilities shared by both models."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


LABELS = ["positive", "negative", "neutral"]


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
	report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
	return {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision": report["macro avg"]["precision"],
		"recall": report["macro avg"]["recall"],
		"f1": report["macro avg"]["f1-score"],
	}


def save_evaluation(y_true, y_pred, output_dir: str | Path, model_name: str) -> dict[str, float]:
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	metrics = calculate_metrics(y_true, y_pred)
	pd.DataFrame([metrics]).assign(model=model_name).to_csv(output_dir / f"{model_name}_metrics.csv", index=False)
	matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
	pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(output_dir / f"{model_name}_confusion_matrix.csv")
	return metrics
