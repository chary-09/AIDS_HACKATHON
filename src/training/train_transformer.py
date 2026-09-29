"""Fine-tune a Hugging Face transformer for aspect sentiment classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.loader import load_processed
from src.evaluation.evaluate import save_evaluation
from src.models.transformer import ABSADataset, create_model, save_model


LABELS = ["positive", "negative", "neutral"]


def _run_epoch(model, loader, optimizer, device, loss_function, training):
    model.train(training)
    total_loss = 0.0
    predictions, labels = [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if training:
            optimizer.zero_grad()
        labels_tensor = batch.pop("labels")
        output = model(**batch)
        loss = loss_function(output.logits, labels_tensor)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels_tensor.size(0)
        predictions.extend(output.logits.argmax(dim=1).detach().cpu().tolist())
        labels.extend(labels_tensor.detach().cpu().tolist())
    return total_loss / max(len(loader.dataset), 1), predictions, labels


def train(args):
    frame = load_processed(args.data)
    frame = frame[frame["sentiment"].isin(LABELS)].reset_index(drop=True)
    train_frame, test_frame = train_test_split(frame, test_size=args.test_size, random_state=42, stratify=frame["sentiment"])
    train_frame, validation_frame = train_test_split(
        train_frame, test_size=args.validation_size, random_state=42, stratify=train_frame["sentiment"]
    )

    tokenizer, model = create_model(args.model_name)
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    datasets = {
        "train": ABSADataset(train_frame, tokenizer, label_to_id, args.max_length),
        "validation": ABSADataset(validation_frame, tokenizer, label_to_id, args.max_length),
        "test": ABSADataset(test_frame, tokenizer, label_to_id, args.max_length),
    }
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train") for name, dataset in datasets.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_weights = None
    if args.class_weights:
        counts = train_frame["sentiment"].value_counts()
        class_weights = torch.tensor(
            [len(train_frame) / (len(LABELS) * max(counts.get(label, 0), 1)) for label in LABELS],
            dtype=torch.float,
            device=device,
        )
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = _run_epoch(model, loaders["train"], optimizer, device, loss_function, training=True)
        with torch.no_grad():
            validation_loss, _, _ = _run_epoch(model, loaders["validation"], optimizer, device, loss_function, training=False)
        print(f"Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f} - validation loss: {validation_loss:.4f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_model(model, tokenizer, args.model_dir)

    with torch.no_grad():
        _, predictions, labels = _run_epoch(model, loaders["test"], optimizer, device, loss_function, training=False)
    metrics = save_evaluation(labels, predictions, args.output_dir, args.run_name)
    print(f"Device: {device}")
    print(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/absa.csv")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--model-dir", default="models/transformer")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--run-name", default="transformer")
    train(parser.parse_args())


if __name__ == "__main__":
    main()