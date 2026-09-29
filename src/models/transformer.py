"""Hugging Face model and Dataset definitions for ABSA classification."""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ABSADataset(Dataset):
	def __init__(self, frame, tokenizer, label_to_id, max_length=128):
		self.texts = frame["text"].tolist()
		self.aspects = frame["aspect"].tolist()
		self.labels = [label_to_id[label] for label in frame["sentiment"]]
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		encoded = self.tokenizer(
			self.texts[index], self.aspects[index], truncation=True,
			padding="max_length", max_length=self.max_length, return_tensors="pt",
		)
		return {key: value.squeeze(0) for key, value in encoded.items()} | {
			"labels": torch.tensor(self.labels[index], dtype=torch.long)
		}


def create_model(model_name="distilbert-base-uncased"):
	labels = ["positive", "negative", "neutral"]
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSequenceClassification.from_pretrained(
		model_name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={label: i for i, label in enumerate(labels)}
	)
	return tokenizer, model


def save_model(model, tokenizer, output_dir="models/transformer"):
	Path(output_dir).mkdir(parents=True, exist_ok=True)
	model.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)
