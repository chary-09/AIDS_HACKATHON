# ABSA Hackathon

Aspect-Based Sentiment Analysis project for restaurant, laptop, MAMS, and hackathon datasets.

## Project Structure

```text
ABSA-Hackathon/
├── data/
│   ├── raw/
│   │   ├── semeval/
│   │   ├── mams/
│   │   └── hackathon/
│   ├── processed/
│   └── final/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── baseline.py
│   │   └── transformer.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   └── evaluate.py
│   └── inference/
│       └── predict.py
├── app/
│   └── app.py
├── models/
├── outputs/
├── requirements.txt
├── README.md
└── .gitignore
```

### Folder Responsibilities

- `data/raw/`: Original downloaded datasets. Do not overwrite these files.
- `data/processed/`: Clean records converted to the common ABSA format.
- `data/final/`: Final train, validation, and test files selected for experiments.
- `notebooks/`: Exploratory analysis, preprocessing experiments, baseline work, and evaluation.
- `src/`: Reusable Python code for loading data, modeling, training, evaluation, and prediction.
- `models/`: Saved model checkpoints and tokenizers.
- `outputs/`: Metrics, predictions, plots, and experiment logs.
- `app/`: The Streamlit demo used for inference.

### Files We Will Create

- `loader.py`: Read each raw dataset and expose a consistent interface.
- `preprocess.py`: Validate records and convert them to the common format.
- `baseline.py`: A simple baseline model for comparison.
- `transformer.py`: Transformer-based ABSA model code.
- `train.py`: Training entry point.
- `evaluate.py`: Metrics and error analysis.
- `predict.py`: Reusable inference functions for the demo.
- `app.py`: Streamlit user interface.
- The four notebooks: One focused notebook for each investigation stage.

## Development Phases

### Phase 1: Understand and Prepare the Data

Explore all datasets, inspect label distributions, convert them to the common format, remove only invalid records, and create train/validation/test splits.

### Phase 2: Train and Evaluate Models

Build a simple baseline, fine-tune a Hugging Face transformer, compare metrics, and inspect aspect-level errors.

### Phase 3: Demo and Final Experiments

Save the best model, connect it to Streamlit, test example sentences, and record final results in `outputs/`.

## Dataset Organization

Place source files here:

- `data/raw/semeval/`: SemEval-2014 Restaurant and Laptop files.
- `data/raw/mams/`: MAMS ABSA files.
- `data/raw/hackathon/`: The hackathon dataset exactly as received.
- `data/processed/`: Normalized files generated from the raw data.

### What the Datasets Contain

- **SemEval-2014 Restaurant**: Reviews with aspect terms such as `food` or `service`, usually with aspect sentiment polarity. Restaurant aspect categories may also be available.
- **SemEval-2014 Laptop**: Laptop reviews with aspect terms and sentiment polarity. Product aspect categories may also be available.
- **MAMS ABSA**: Multi-aspect sentences. A single sentence can contain several aspects with different sentiment labels, so each aspect occurrence must remain a separate record.
- **Hackathon dataset**: Use its provided text, aspect, category, and sentiment fields. First document its actual schema in the preprocessing notebook.

### Important Labels

The main target is aspect sentiment polarity: `positive`, `negative`, `neutral`, and `conflict` when supplied. Preserve aspect terms and their character positions whenever available. Preserve category labels, dataset names, IDs, and any original labels as metadata; do not discard them during normalization.

### Common Processed Format

Store one row per aspect occurrence. Recommended columns:

| Column | Meaning |
| --- | --- |
| `id` | Stable unique record ID |
| `text` | Full original sentence or review sentence |
| `aspect` | Exact aspect text from the source |
| `aspect_start` | Character start position, if available |
| `aspect_end` | Character end position, if available |
| `category` | Aspect category, if available |
| `sentiment` | Normalized polarity label |
| `original_sentiment` | Original source label |
| `dataset` | `semeval_restaurant`, `semeval_laptop`, `mams`, or `hackathon` |
| `split` | Original split, if supplied |
| `source_id` | Original dataset record ID |

The minimum required fields are `id`, `text`, `aspect`, `sentiment`, and `dataset`. Missing optional values should be empty, not invented.

### Normalization and Validation Rules

1. Convert every aspect occurrence into its own row, even when several aspects come from the same sentence.
2. Normalize equivalent polarity names to lowercase `positive`, `negative`, `neutral`, or `conflict`.
3. Keep `original_sentiment` and all source metadata for traceability.
4. Keep raw files unchanged. Write cleaned data only under `data/processed/`.
5. Reject or quarantine records with missing text, missing aspect, unusable sentiment, or invalid aspect offsets. Record the reason in a validation report.
6. If offsets are absent but text and aspect are valid, keep the record and leave `aspect_start` and `aspect_end` empty.
7. Do not silently guess an aspect, sentiment, category, or offset. Preserve uncertain records separately for review.

Preprocessing code will be added after the raw schemas are inspected and this common format is confirmed.

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The project intentionally contains no training or preprocessing implementation yet.