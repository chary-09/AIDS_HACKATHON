"""Generate and validate official Hackathon Submission file based on test.csv."""

import argparse
from pathlib import Path
import pandas as pd

from src.inference.predict import get_pipeline


def generate_submission(
    test_path: str = "data/raw/hackathon/dataset/test/test.csv",
    sample_sub_path: str = "data/raw/hackathon/dataset/test/sample_submission.csv",
    output_path: str = "outputs/submission.csv"
):
    print("=== GENERATING OFFICIAL HACKATHON SUBMISSION ===")
    t_path = Path(test_path)
    s_path = Path(sample_sub_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = get_pipeline()

    if not t_path.exists():
        print(f"Warning: Test file not found at {t_path}. Creating fallback submission.")
        return

    df_test = pd.read_csv(t_path)
    print(f"Loaded {len(df_test)} test examples from {t_path}")

    submission_rows = []

    for idx, row in df_test.iterrows():
        rec_id = str(row.get("id", f"ABSA_TEST_{idx+1:04d}"))
        text = str(row.get("text", "")).strip()
        aspect = str(row.get("aspect_term", "")).strip()

        # If aspect term is provided in test row, classify sentiment directly for that aspect context
        if aspect and text:
            sentiment, _ = pipeline.classify_sentiment(text, aspect)
        else:
            preds = pipeline.predict(text)
            if preds:
                sentiment = preds[0]["sentiment"]
            else:
                sentiment = "neutral"

        # Format label to title case matching sample_submission.csv (Positive, Negative, Neutral)
        formatted_sentiment = sentiment.capitalize()

        submission_rows.append({
            "id": rec_id,
            "prediction": formatted_sentiment
        })

    df_sub = pd.DataFrame(submission_rows)

    # Submission validation rules
    print("\n--- Validating Submission File ---")
    assert not df_sub.empty, "Error: Submission DataFrame is empty!"
    assert list(df_sub.columns) == ["id", "prediction"], f"Error: Column header mismatch! Found {list(df_sub.columns)}"
    assert df_sub["prediction"].isnull().sum() == 0, "Error: Found NaN values in predictions!"
    assert df_sub["prediction"].isin(["Positive", "Negative", "Neutral"]).all(), "Error: Invalid sentiment labels found!"

    if s_path.exists():
        df_sample = pd.read_csv(s_path)
        assert len(df_sub) == len(df_sample), f"Row count mismatch! Generated {len(df_sub)} vs Sample {len(df_sample)}"
        assert (df_sub["id"] == df_sample["id"]).all(), "Error: Record IDs do not match sample submission!"
        print("✅ Exact row count and ID sequence match with sample_submission.csv!")

    df_sub.to_csv(out_path, index=False)
    print(f"✅ Official submission file successfully generated and saved to: {out_path}")
    print("\nFirst 10 Rows of Submission File:")
    print(df_sub.head(10))
    print("\nClass Distribution:")
    print(df_sub["prediction"].value_counts())

    return df_sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="data/raw/hackathon/dataset/test/test.csv")
    parser.add_argument("--output", default="outputs/submission.csv")
    args = parser.parse_args()
    generate_submission(args.test_file, output_path=args.output)
