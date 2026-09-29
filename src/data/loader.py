"""Load the normalized ABSA dataset."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"id", "text", "aspect", "sentiment"}


def load_processed(path: str | Path = "data/processed/absa.csv") -> pd.DataFrame:
	frame = pd.read_csv(path, keep_default_na=False)
	missing = REQUIRED_COLUMNS.difference(frame.columns)
	if missing:
		raise ValueError(f"Processed dataset is missing columns: {sorted(missing)}")
	return frame
