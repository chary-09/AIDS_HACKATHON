"""Evaluate Aspect Extractor performance on token & span level metrics."""

import json
from pathlib import Path
from src.models.aspect_extractor import BIOAspectExtractor


def evaluate_aspect_extractor(
    dataset_path: str = "data/processed/aspect_extraction_dataset.json",
    model_dir: str = "models/aspect_extractor"
) -> dict:
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    extractor = BIOAspectExtractor()
    model_path = Path(model_dir)
    if model_path.exists():
        extractor.load(model_dir)
    else:
        # Populate learned aspects if un-trained checkpoint
        for r in records:
            for a in r["aspects"]:
                extractor.learned_aspects.add(a["aspect"].lower().strip())

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    samples_results = []

    for item in records:
        text = item["text"]
        true_aspects = set(a["aspect"].lower().strip() for a in item["aspects"])
        
        preds = extractor.extract_aspects(text)
        pred_aspects = set(p["aspect"].lower().strip() for p in preds)

        tp = len(true_aspects.intersection(pred_aspects))
        fp = len(pred_aspects - true_aspects)
        fn = len(true_aspects - pred_aspects)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

        samples_results.append({
            "text": text,
            "ground_truth": list(true_aspects),
            "predicted": list(pred_aspects),
            "tp": tp,
            "fp": fp,
            "fn": fn
        })

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "aspect_precision": precision,
        "aspect_recall": recall,
        "aspect_f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

    print("=== ASPECT EXTRACTION EVALUATION ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")

    return metrics, samples_results


if __name__ == "__main__":
    evaluate_aspect_extractor()
