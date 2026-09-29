"""Train the Aspect Extractor model on annotated dataset aspect terms."""

import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW

from src.models.aspect_extractor import BIOAspectExtractor, LABEL2ID, ID2LABEL
from src.data.populate_datasets import populate_all


class AspectExtractionDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.labels = []

        extractor = BIOAspectExtractor()

        for item in records:
            text = item["text"]
            aspects = item["aspects"]
            words, bio_tags = extractor.convert_sentence_to_bio(text, aspects)

            # Tokenize words with subwords
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            word_ids = encoding.word_ids(batch_index=0)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(LABEL2ID.get(bio_tags[word_idx], LABEL2ID["O"]))
                else:
                    tag = bio_tags[word_idx]
                    if tag == "B-ASPECT":
                        tag = "I-ASPECT"
                    label_ids.append(LABEL2ID.get(tag, LABEL2ID["O"]))
                previous_word_idx = word_idx

            self.inputs.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            })
            self.labels.append(torch.tensor(label_ids, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {k: v for k, v in self.inputs[idx].items()}
        item["labels"] = self.labels[idx]
        return item


def train_aspect_extractor(
    dataset_path: str = "data/processed/aspect_extraction_dataset.json",
    model_dir: str = "models/aspect_extractor",
    epochs: int = 5,
    lr: float = 3e-5,
    batch_size: int = 8
):
    path = Path(dataset_path)
    if not path.exists():
        populate_all()

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    extractor = BIOAspectExtractor()
    
    # Store learned aspect terms vocabulary for backup matchers
    for r in records:
        for a in r.get("aspects", []):
            extractor.learned_aspects.add(a["aspect"].lower().strip())

    ds = AspectExtractionDataset(records, extractor.tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(extractor.model.parameters(), lr=lr)

    extractor.model.train()
    print(f"Training Aspect Extractor on {len(records)} sentences for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(extractor.device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = extractor.model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    extractor.save(model_dir)
    print(f"Aspect Extractor model saved to {model_dir}")
    return extractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/aspect_extraction_dataset.json")
    parser.add_argument("--model-dir", default="models/aspect_extractor")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    train_aspect_extractor(args.dataset, args.model_dir, epochs=args.epochs)
