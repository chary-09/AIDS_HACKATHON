"""Benchmark Comparison: Our Model vs Outside / Baseline Models.

Compares:
1. Outside Model 1: VADER / Lexicon-Based Sentiment (Global Sentence Level)
2. Outside Model 2: Standard Unigram TF-IDF (Aspect-Blind)
3. Outside Model 3: Generic Review-Level Zero-Shot Sentiment
4. Our Model: Context-Aware Aspect-Targeted Sentiment Model (Trained on Hackathon Dataset)
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.baseline import build_accurate_model, make_aspect_aware_input
from src.data.populate_datasets import populate_all


def run_benchmark():
    print("=" * 70)
    print("       ABSA BENCHMARK COMPARISON: OUR MODEL VS OUTSIDE MODELS        ")
    print("=" * 70)

    # 1. Load data
    train_file = Path("data/raw/hackathon/dataset/train/train.csv")
    if not train_file.exists():
        populate_all()
        df = pd.read_csv("data/processed/absa.csv")
    else:
        df = pd.read_csv(train_file)
        # Normalize labels
        df["sentiment"] = df["label"].astype(str).str.lower().str.strip()
        df["aspect"] = df["aspect_term"]

    df = df[df["sentiment"].isin(["positive", "negative", "neutral"])].reset_index(drop=True)
    print(f"Loaded {len(df)} total annotated samples for evaluation.")

    # Train / Val Split (80% Train, 20% Validation)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])
    y_val = val_df["sentiment"].tolist()
    labels = ["positive", "negative", "neutral"]

    results = []

    # -------------------------------------------------------------
    # OUTSIDE MODEL 1: Lexicon-Based Sentiment (Aspect-Blind Global)
    # -------------------------------------------------------------
    print("\nEvaluating Outside Model 1 (Lexicon Sentiment Analyzer)...")
    pos_words = {"good", "great", "excellent", "best", "delicious", "nice", "friendly", "love", "fresh", "crisp", "amazing", "wonderful", "reasonable", "fast", "top"}
    neg_words = {"bad", "poor", "terrible", "worst", "slow", "horrible", "cramped", "expensive", "noisy", "awful", "cold", "bland", "dirty", "unhelpful", "rude"}
    
    y_pred_outside1 = []
    for text in val_df["text"]:
        txt_low = str(text).lower()
        p_c = sum(1 for w in pos_words if w in txt_low)
        n_c = sum(1 for w in neg_words if w in txt_low)
        if p_c > n_c:
            y_pred_outside1.append("positive")
        elif n_c > p_c:
            y_pred_outside1.append("negative")
        else:
            y_pred_outside1.append("neutral")

    acc1 = accuracy_score(y_val, y_pred_outside1)
    p1, r1, f1_1, _ = precision_recall_fscore_support(y_val, y_pred_outside1, labels=labels, average="macro", zero_division=0)
    results.append({
        "Model Name": "Outside Model 1 (Lexicon Rule-Based)",
        "Type": "Unsupervised / Aspect-Blind",
        "Accuracy": round(acc1 * 100, 2),
        "Macro Precision": round(p1 * 100, 2),
        "Macro Recall": round(r1 * 100, 2),
        "Macro F1": round(f1_1 * 100, 2),
    })

    # -------------------------------------------------------------
    # OUTSIDE MODEL 2: Plain Unigram TF-IDF (Aspect-Blind)
    # -------------------------------------------------------------
    print("Evaluating Outside Model 2 (Standard Unigram TF-IDF)...")
    pipe_unigram = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe_unigram.fit(train_df["text"], train_df["sentiment"])
    y_pred_outside2 = pipe_unigram.predict(val_df["text"])

    acc2 = accuracy_score(y_val, y_pred_outside2)
    p2, r2, f1_2, _ = precision_recall_fscore_support(y_val, y_pred_outside2, labels=labels, average="macro", zero_division=0)
    results.append({
        "Model Name": "Outside Model 2 (Plain Unigram TF-IDF)",
        "Type": "Standard Classifier / Aspect-Blind",
        "Accuracy": round(acc2 * 100, 2),
        "Macro Precision": round(p2 * 100, 2),
        "Macro Recall": round(r2 * 100, 2),
        "Macro F1": round(f1_2 * 100, 2),
    })

    # -------------------------------------------------------------
    # OUTSIDE MODEL 3: Generic Zero-Shot Sentence Classifier
    # -------------------------------------------------------------
    print("Evaluating Outside Model 3 (Generic Review Classifier)...")
    pipe_generic = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe_generic.fit(train_df["text"], train_df["sentiment"])
    y_pred_outside3 = pipe_generic.predict(val_df["text"])

    acc3 = accuracy_score(y_val, y_pred_outside3)
    p3, r3, f1_3, _ = precision_recall_fscore_support(y_val, y_pred_outside3, labels=labels, average="macro", zero_division=0)
    results.append({
        "Model Name": "Outside Model 3 (Generic Review Classifier)",
        "Type": "Review-Level Bigram Classifier",
        "Accuracy": round(acc3 * 100, 2),
        "Macro Precision": round(p3 * 100, 2),
        "Macro Recall": round(r3 * 100, 2),
        "Macro F1": round(f1_3 * 100, 2),
    })

    # -------------------------------------------------------------
    # OUR MODEL: Context-Aware Aspect-Targeted Model
    # -------------------------------------------------------------
    print("Training and Evaluating OUR MODEL (Context-Aware ABSA Model)...")
    ctx_train = train_df.get("context_with_aspect_tag")
    ctx_val = val_df.get("context_with_aspect_tag")

    X_train = make_aspect_aware_input(train_df["text"], train_df["aspect"], ctx_train)
    X_val = make_aspect_aware_input(val_df["text"], val_df["aspect"], ctx_val)

    our_model = build_accurate_model()
    our_model.fit(X_train, train_df["sentiment"])
    y_pred_our = our_model.predict(X_val)

    acc_our = accuracy_score(y_val, y_pred_our)
    p_our, r_our, f1_our, _ = precision_recall_fscore_support(y_val, y_pred_our, labels=labels, average="macro", zero_division=0)
    results.append({
        "Model Name": "⭐ OUR MODEL (Aspect-Targeted ABSA)",
        "Type": "Context-Aware Feature Union (Trained)",
        "Accuracy": round(acc_our * 100, 2),
        "Macro Precision": round(p_our * 100, 2),
        "Macro Recall": round(r_our * 100, 2),
        "Macro F1": round(f1_our * 100, 2),
    })

    # Save trained accurate model
    Path("models").mkdir(parents=True, exist_ok=True)
    import pickle
    with open("models/baseline.pkl", "wb") as f:
        pickle.dump(our_model, f)
    print("Saved high-accuracy model to models/baseline.pkl")

    # Save benchmark table
    df_res = pd.DataFrame(results)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / "model_comparison.csv", index=False)
    with open(out_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(df_res.to_string(index=False))
    print("=" * 70)
    print(f"Results saved to {out_dir / 'model_comparison.csv'}")

    return df_res


if __name__ == "__main__":
    run_benchmark()
