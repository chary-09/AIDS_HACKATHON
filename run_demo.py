"""Convenience runner script for dataset processing, model training, validation, submission, and Streamlit."""

import sys
import subprocess
from pathlib import Path


def print_banner():
    print("=" * 65)
    print("   ASPECT-BASED SENTIMENT ANALYSIS (ABSA) HACKATHON RUNNER   ")
    print("=" * 65)


def run_populate():
    print("\n--- Populating Official Hackathon Dataset ---")
    from src.data.populate_datasets import populate_all
    populate_all()


def run_train():
    print("\n--- Training Baseline Sentiment Classifier Model ---")
    from src.training.train import main as train_baseline
    train_baseline()


def run_pipeline_test():
    print("\n--- Running ABSA Inference Test (10 Examples) ---")
    from src.inference.predict import predict_absa
    examples = [
        "The camera is excellent but the battery life is poor.",
        "Food was delicious and the service was top notch.",
        "The atmosphere is nice, but the pasta was cold and bland.",
        "Screen resolution is sharp and vivid.",
        "Waitstaff was polite, but the food arrived very late.",
        "Audio clarity is decent, but bass response is completely lacking.",
        "Fast boot speed, but the fan noise is irritating.",
        "Friendly manager, average drinks, terrible ambiance.",
        "Customer support was unhelpful and slow to respond.",
        "The touchscreen responsiveness is laggy."
    ]
    for idx, text in enumerate(examples, 1):
        res = predict_absa(text)
        print(f"\n[{idx}] Review: '{text}'")
        print(f"    Predictions: {res}")


def run_validation():
    print("\n--- Running Full Pipeline Validation Suite ---")
    from src.evaluation.validate_pipeline import run_validation as validate
    validate()


def run_submission():
    print("\n--- Generating Official Hackathon Submission File ---")
    from src.inference.submission import generate_submission
    generate_submission()


def run_benchmark():
    print("\n--- Running Benchmark Comparison: Our Model vs Outside Models ---")
    from src.evaluation.benchmark_comparison import run_benchmark as bench
    bench()


def run_streamlit():
    print("\n--- Launching Streamlit Web Application ---")
    cmd = [sys.executable, "-m", "streamlit", "run", "app/app.py"]
    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"Error launching streamlit: {e}")
        print("Please ensure streamlit is installed: pip install streamlit")


if __name__ == "__main__":
    print_banner()
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["populate", "data"]:
            run_populate()
        elif arg in ["train"]:
            run_train()
        elif arg in ["benchmark", "compare", "models"]:
            run_benchmark()
        elif arg in ["test", "predict"]:
            run_pipeline_test()
        elif arg in ["validate", "eval"]:
            run_validation()
        elif arg in ["submission", "sub"]:
            run_submission()
        elif arg in ["app", "streamlit", "ui"]:
            run_streamlit()
    else:
        # Complete end-to-end execution pipeline when run directly
        run_populate()
        run_train()
        run_benchmark()
        run_pipeline_test()
        run_validation()
        run_submission()

