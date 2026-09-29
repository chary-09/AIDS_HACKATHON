"""High-Accuracy Aspect-Targeted Sentiment Analysis Model."""

import re
from pathlib import Path
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression


def make_aspect_aware_input(texts, aspects, contexts_with_tag=None) -> list[str]:
    """Construct localized aspect-context features for high classification accuracy."""
    inputs = []
    contexts_list = list(contexts_with_tag) if contexts_with_tag is not None else [None] * len(texts)

    for text, aspect, tagged_ctx in zip(texts, aspects, contexts_list):
        text_str = str(text) if pd.notnull(text) else ""
        aspect_str = str(aspect).strip() if pd.notnull(aspect) else ""
        
        # Local window around aspect term (40 chars left & right)
        idx = text_str.lower().find(aspect_str.lower())
        if idx != -1:
            start = max(0, idx - 45)
            end = min(len(text_str), idx + len(aspect_str) + 45)
            local_window = text_str[start:end]
        else:
            local_window = text_str

        # Tagged representation replacing aspect with marker
        if tagged_ctx is not None and pd.notnull(tagged_ctx) and "$T$" in str(tagged_ctx):
            tagged = str(tagged_ctx).replace("$T$", f"__ASPECT_{aspect_str}__")
        else:
            tagged = text_str.replace(aspect_str, f"__ASPECT_{aspect_str}__")

        # Structured composite feature string
        combined = (
            f"__ASPECT_{aspect_str}__ "
            f"__LOCAL_START__ {local_window} __LOCAL_END__ "
            f"__TAGGED_START__ {tagged} __TAGGED_END__ "
            f"{text_str}"
        )
        inputs.append(combined)
    return inputs


def make_input(texts, aspects):
    """Backward compatible input helper."""
    return make_aspect_aware_input(texts, aspects)


def build_accurate_model() -> Pipeline:
    """Build multi-ngram feature union with balanced logistic regression for maximum accuracy."""
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b|__\w+__",
            max_features=30000
        )),
        ("char_tfidf", TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer="char_wb",
            min_df=1,
            sublinear_tf=True,
            max_features=20000
        ))
    ])

    classifier = LogisticRegression(
        C=3.0,
        max_iter=1500,
        class_weight="balanced",
        solver="lbfgs"
    )

    return Pipeline([
        ("features", features),
        ("classifier", classifier)
    ])


def build_baseline(class_weight="balanced") -> Pipeline:
    return build_accurate_model()


def save_baseline(model: Pipeline, path: str | Path = "models/baseline.pkl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as handle:
        pickle.dump(model, handle)
