"""Normalize raw ABSA files into one aspect-level CSV format."""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


LABEL_MAP = {
	"positive": "positive",
	"pos": "positive",
	"1": "positive",
	"negative": "negative",
	"neg": "negative",
	"-1": "negative",
	"neutral": "neutral",
	"neu": "neutral",
	"0": "neutral",
}
REQUIRED_COLUMNS = ["id", "text", "aspect", "sentiment"]


def _clean_key(value: Any) -> str:
	return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _as_records(value: Any) -> list[dict[str, Any]]:
	if isinstance(value, list):
		return [item for item in value if isinstance(item, dict)]
	if isinstance(value, dict):
		for key in ("data", "records", "examples", "sentences", "annotations"):
			if isinstance(value.get(key), list):
				return _as_records(value[key])
		return [value]
	return []


def _read_delimited(path: Path) -> list[dict[str, Any]]:
	with path.open("r", encoding="utf-8-sig", newline="") as handle:
		sample = handle.read(4096)
		handle.seek(0)
		dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|\n")
		return list(csv.DictReader(handle, dialect=dialect))


def _read_json(path: Path) -> list[dict[str, Any]]:
	with path.open("r", encoding="utf-8-sig") as handle:
		content = handle.read().strip()
	try:
		return _as_records(json.loads(content))
	except json.JSONDecodeError:
		return [json.loads(line) for line in content.splitlines() if line.strip()]


def _xml_records(path: Path) -> list[dict[str, Any]]:
	root = ET.parse(path).getroot()
	records: list[dict[str, Any]] = []
	for sentence in root.iter():
		if not sentence.tag.lower().endswith("sentence"):
			continue
		text = sentence.attrib.get("text") or "".join(sentence.itertext()).strip()
		for opinion in sentence.iter():
			if not opinion.tag.lower().endswith("opinion"):
				continue
			record = dict(opinion.attrib)
			record["text"] = text
			record["source_id"] = sentence.attrib.get("id", "")
			records.append(record)
	return records


def read_raw_file(path: Path) -> list[dict[str, Any]]:
	suffix = path.suffix.lower()
	if suffix in {".csv", ".tsv", ".txt"}:
		return _read_delimited(path)
	if suffix in {".json", ".jsonl"}:
		return _read_json(path)
	if suffix == ".xml":
		return _xml_records(path)
	raise ValueError(f"Unsupported file type: {path.name}")


def _find_value(record: dict[str, Any], aliases: set[str]) -> Any:
	for key, value in record.items():
		if _clean_key(key) in aliases:
			return value
	return None


def normalize_sentiment(value: Any) -> str | None:
	if value is None or pd.isna(value):
		return None
	return LABEL_MAP.get(str(value).strip().lower())


def normalize_record(record: dict[str, Any], dataset: str, source_id: str) -> tuple[dict[str, Any] | None, str | None]:
	text = _find_value(record, {"text", "sentence", "review", "content"})
	aspect = _find_value(record, {"aspect", "aspectterm", "term", "target", "category"})
	sentiment_value = _find_value(record, {"sentiment", "polarity", "label", "情感"})
	if text is None or not str(text).strip():
		return None, "missing_text"
	if aspect is None or not str(aspect).strip():
		return None, "missing_aspect"
	sentiment = normalize_sentiment(sentiment_value)
	if sentiment is None:
		return None, "missing_or_unknown_sentiment"

	record_id = _find_value(record, {"id", "recordid", "sourceid"}) or source_id
	return {
		"id": f"{dataset}:{record_id}",
		"text": str(text).strip(),
		"aspect": str(aspect).strip(),
		"sentiment": sentiment,
	}, None


def process_directory(raw_root: Path, output_path: Path, rejected_path: Path) -> pd.DataFrame:
	rows: list[dict[str, str]] = []
	rejected: list[dict[str, str]] = []
	for path in sorted(raw_root.rglob("*")):
		if not path.is_file() or path.name.startswith("."):
			continue
		dataset = path.parent.name
		try:
			records = read_raw_file(path)
		except (OSError, ValueError, json.JSONDecodeError, ET.ParseError) as error:
			rejected.append({"file": str(path), "reason": f"read_error:{error}"})
			continue
		for index, record in enumerate(records):
			normalized, reason = normalize_record(record, dataset, f"{path.stem}:{index}")
			if normalized:
				rows.append(normalized)
			else:
				rejected.append({"file": str(path), "record": str(index), "reason": reason or "invalid"})

	frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
	if not frame.empty:
		frame = frame.drop_duplicates(subset=["text", "aspect", "sentiment"], keep="first")
		frame = frame.reset_index(drop=True)
		frame["id"] = [f"absa:{index:07d}" for index in range(len(frame))]
	output_path.parent.mkdir(parents=True, exist_ok=True)
	rejected_path.parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(output_path, index=False)
	pd.DataFrame(rejected).to_csv(rejected_path, index=False)
	return frame


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
	parser.add_argument("--output", type=Path, default=Path("data/processed/absa.csv"))
	parser.add_argument("--rejected", type=Path, default=Path("data/processed/rejected.csv"))
	args = parser.parse_args()
	frame = process_directory(args.raw_dir, args.output, args.rejected)
	print(f"Saved {len(frame)} valid records to {args.output}")


if __name__ == "__main__":
	main()
