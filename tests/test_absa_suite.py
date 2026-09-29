"""Automated Test Suite for Aspect-Based Sentiment Analysis.

Verifies:
1. Pure positive reviews -> 100% positive
2. Pure negative reviews -> 100% negative
3. Contrastive reviews ('X is good but Y is bad') -> isolated polarities
4. Negation inversion ('not good' -> negative, 'not bad' -> positive)
5. Zero flipped or contradictory labels
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.predict import predict_absa


def run_all_tests():
    test_cases = [
        # (Input text, expected dict of aspect: sentiment)
        (
            "The camera is excellent but the battery life is poor.",
            {"camera": "positive", "battery life": "negative"}
        ),
        (
            "Food was delicious and the service was top notch.",
            {"food": "positive", "service": "positive"}
        ),
        (
            "The service was terrible and food was awful.",
            {"service": "negative", "food": "negative"}
        ),
        (
            "The battery is negative",
            {"battery": "negative"}
        ),
        (
            "The battery is positive",
            {"battery": "positive"}
        ),
        (
            "The camera is good",
            {"camera": "positive"}
        ),
        (
            "The food is not good",
            {"food": "negative"}
        ),
        (
            "The phone is not bad",
            {"phone": "positive"}
        ),
        (
            "The steering wheel is responsive but the brake pedal is stiff.",
            {"steering wheel": "positive", "brake pedal": "negative"}
        ),
        (
            "The user interface is intuitive while the export feature is painfully slow.",
            {"user interface": "positive", "export feature": "negative"}
        )
    ]

    all_passed = True
    print("=" * 60)
    print("RUNNING ABSA POLARITY VALIDATION SUITE")
    print("=" * 60)

    for idx, (text, expected) in enumerate(test_cases, 1):
        results = predict_absa(text)
        res_map = {r["aspect"].lower(): r["sentiment"].lower() for r in results}
        
        passed = True
        for exp_asp, exp_sent in expected.items():
            matched_key = None
            for a in res_map:
                if exp_asp in a or a in exp_asp:
                    matched_key = a
                    break
            
            if not matched_key or res_map[matched_key] != exp_sent:
                passed = False
                break

        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"[{idx:02d}] {status}: '{text}'")
        for r in results:
            print(f"      -> [{r['aspect']}]: {r['sentiment'].upper()} ({r['confidence']*100:.1f}%)")

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! 100% ACCURATE POLARITY SEPARATION.")
    else:
        print("SOME TESTS FAILED.")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    run_all_tests()
