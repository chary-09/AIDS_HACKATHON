"""Create a comparison table from metrics files produced by real runs."""

from pathlib import Path

import pandas as pd


def build_comparison(output_dir="outputs", output_file="outputs/model_comparison.csv"):
    rows = []
    for metrics_file in sorted(Path(output_dir).glob("*_metrics.csv")):
        frame = pd.read_csv(metrics_file)
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        rows.append({
            "Model": row.get("model", metrics_file.stem.replace("_metrics", "")),
            "Accuracy": row.get("accuracy"),
            "Precision": row.get("precision"),
            "Recall": row.get("recall"),
            "F1": row.get("f1"),
        })
    comparison = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_file, index=False)
    if comparison.empty:
        print("No completed evaluation files found; comparison table is empty.")
    else:
        print(comparison.to_string(index=False))
    return comparison


if __name__ == "__main__":
    build_comparison()