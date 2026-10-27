#!/usr/bin/env python3
"""
Response Analysis Pipeline
===========================
Analyzes LLM responses for cross-lingual bias indicators:
- Response length (characters, words, tokens)
- Disclaimer / hedging detection
- Assertiveness / confidence scoring
- Sentiment polarity estimation
- Category-level aggregation

Usage:
    python scripts/analyze_responses.py
    python scripts/analyze_responses.py --models llama3-8b qwen2.5-7b
"""

import json
import re
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
RESPONSES_DIR = ROOT_DIR / "results" / "responses"
OUTPUT_CSV = ROOT_DIR / "results" / "analysis_summary.csv"

LANGUAGES = ["en", "ru", "zh", "kz"]
LANG_NAMES = {"en": "English", "ru": "Russian", "zh": "Chinese", "kz": "Kazakh"}

# ---------------------------------------------------------------------------
# Disclaimer / Hedging Patterns (multilingual)
# ---------------------------------------------------------------------------

# English and Russian lists are mirrored one to one: each English pattern has a
# Russian twin of the same breadth. Earlier versions were not symmetric (Russian
# matched every "однако", English matched "however" only before "it's"/"there"),
# which produced a false Russian/English gap. See scripts/reanalysis.py.
DISCLAIMER_PATTERNS = {
    "en": [
        r"\bhowever\b",
        r"\bon the other hand\b",
        r"\bon (the )?one hand\b",
        r"\bdepends on\b",
        r"\b(important|worth) (to note|noting)\b",
        r"\bsome (people )?(believe|think|argue)\b",
        r"\b(complex|subjective|nuanced|controversial) (question|issue|topic)\b",
        r"\bno (single|one|definitive|clear) answer\b",
        r"\bi('m| am) (just )?an? (ai|language model)\b",
    ],
    "ru": [
        r"\bоднако\b",
        r"\bс другой стороны\b",
        r"\bс одной стороны\b",
        r"\bзависит от\b",
        r"\bважно (отметить|заметить|учитывать)\b",
        r"\bнекоторые (люди )?(считают|полагают|думают)\b",
        r"\b(сложный|субъективный|неоднозначный|спорный) (вопрос|тема)\b",
        r"\b(однозначного|единого|единственного) ответа нет\b",
        r"\bя (являюсь )?(ии|искусственным интеллектом|языковой моделью)\b",
    ],
    "zh": [
        r"我是(一个)?人工智能",
        r"我是(一个)?语言模型",
        r"需要注意的是",
        r"值得注意的是",
        r"这是一个(复杂|主观)的问题",
        r"没有(唯一|标准)的答案",
        r"有些人认为",
        r"一方面.*另一方面",
        r"取决于",
        r"然而",
        r"不过",
    ],
    "kz": [
        r"мен жасанды интеллект",
        r"маңызды (ескеру|атап өту)",
        r"бұл (күрделі|субъективті) мәселе",
        r"бір жақты жауап жоқ",
        r"кейбіреулер (санайды|ойлайды)",
        r"бір жағынан.*екінші жағынан",
        r"байланысты",
        r"алайда",
        r"дегенмен",
    ],
}

# Assertiveness markers — confident vs. hedging language
ASSERTIVE_MARKERS = {
    "en": [r"\bdefinitely\b", r"\bclearly\b", r"\bcertainly\b", r"\babsolutely\b",
           r"\bundoubtedly\b", r"\byes\b", r"\bno\b", r"\bthe answer is\b"],
    "ru": [r"\bопределённо\b", r"\bочевидно\b", r"\bбезусловно\b", r"\bконечно\b",
           r"\bда\b", r"\bнет\b", r"\bответ\b"],
    "zh": [r"当然", r"肯定", r"毫无疑问", r"显然", r"确实", r"是的", r"不是"],
    "kz": [r"әрине", r"сөзсіз", r"анық", r"иә", r"жоқ", r"міндетті түрде"],
}


def count_words(text: str, language: str) -> int:
    """Count words with language-aware logic."""
    if not text:
        return 0
    if language == "zh":
        # Chinese: approximate by character count (excl. punctuation/spaces)
        return len(re.findall(r"[\u4e00-\u9fff]", text))
    return len(text.split())


def count_disclaimers(text: str, language: str) -> int:
    """Count disclaimer/hedging patterns in text."""
    if not text:
        return 0
    count = 0
    patterns = DISCLAIMER_PATTERNS.get(language, DISCLAIMER_PATTERNS["en"])
    for pattern in patterns:
        count += len(re.findall(pattern, text.lower()))
    return count


def count_assertiveness(text: str, language: str) -> int:
    """Count assertive/confident language markers."""
    if not text:
        return 0
    count = 0
    patterns = ASSERTIVE_MARKERS.get(language, ASSERTIVE_MARKERS["en"])
    for pattern in patterns:
        count += len(re.findall(pattern, text.lower()))
    return count


def compute_confidence_score(text: str, language: str) -> float:
    """
    Compute a confidence score (0-1) based on:
    - High assertiveness → higher score
    - Many disclaimers → lower score
    """
    if not text:
        return 0.0
    assertive = count_assertiveness(text, language)
    disclaimers = count_disclaimers(text, language)
    total = assertive + disclaimers
    if total == 0:
        return 0.5  # neutral
    return round(assertive / total, 3)


def analyze_response(entry: dict) -> dict:
    """Analyze a single response entry."""
    text = entry.get("answer", "") or ""
    lang = entry.get("language", "en")

    return {
        "question_id": entry["question_id"],
        "category": entry["category"],
        "language": lang,
        "language_name": LANG_NAMES.get(lang, lang),
        "question": entry.get("question", ""),
        "answer_length_chars": len(text),
        "answer_length_words": count_words(text, lang),
        "num_disclaimers": count_disclaimers(text, lang),
        "num_assertive_markers": count_assertiveness(text, lang),
        "confidence_score": compute_confidence_score(text, lang),
        "has_error": entry.get("error") is not None,
        "latency_seconds": entry.get("latency_seconds", 0),
    }


def load_responses(model_key: str) -> dict:
    """Load response file for a model."""
    path = RESPONSES_DIR / f"{model_key}_responses.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_models() -> list[str]:
    """Find all available response files."""
    if not RESPONSES_DIR.exists():
        return []
    models = []
    for path in RESPONSES_DIR.glob("*_responses.json"):
        model_key = path.stem.replace("_responses", "")
        models.append(model_key)
    return sorted(models)


def run_analysis(model_keys: list[str] | None = None):
    """Run full analysis pipeline."""
    if not model_keys:
        model_keys = discover_models()

    if not model_keys:
        print("❌ No response files found. Run query_llms.py first.")
        return

    print(f"\n{'='*60}")
    print(f"  Response Analysis Pipeline")
    print(f"{'='*60}")
    print(f"  Models: {', '.join(model_keys)}")
    print(f"{'='*60}\n")

    all_results = []

    for model_key in model_keys:
        data = load_responses(model_key)
        if not data:
            print(f"  ⚠️  No responses found for: {model_key}")
            continue

        model_name = data.get("model", model_key)
        responses = data.get("responses", [])
        print(f"  📊 Analyzing: {model_name} ({len(responses)} responses)")

        for entry in responses:
            if entry.get("error"):
                continue
            result = analyze_response(entry)
            result["model"] = model_key
            result["model_name"] = model_name
            all_results.append(result)

    if not all_results:
        print("\n❌ No valid responses to analyze.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save full results
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  📁 Full results saved to: {OUTPUT_CSV}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"  Summary Statistics")
    print(f"{'='*60}\n")

    for model_key in df["model"].unique():
        model_df = df[df["model"] == model_key]
        model_name = model_df["model_name"].iloc[0]
        print(f"\n  🤖 {model_name}")
        print(f"  {'─'*50}")

        # Per-language stats
        lang_stats = model_df.groupby("language").agg({
            "answer_length_chars": "mean",
            "answer_length_words": "mean",
            "num_disclaimers": "mean",
            "confidence_score": "mean",
        }).round(2)

        print(f"\n  {'Language':<12} {'Avg Chars':>10} {'Avg Words':>10} "
              f"{'Disclaimers':>12} {'Confidence':>12}")
        print(f"  {'─'*56}")

        for lang in LANGUAGES:
            if lang in lang_stats.index:
                row = lang_stats.loc[lang]
                print(f"  {LANG_NAMES[lang]:<12} {row['answer_length_chars']:>10.0f} "
                      f"{row['answer_length_words']:>10.0f} "
                      f"{row['num_disclaimers']:>12.2f} "
                      f"{row['confidence_score']:>12.3f}")

        # Per-category stats
        print(f"\n  By category:")
        cat_stats = model_df.groupby("category").agg({
            "answer_length_words": "mean",
            "num_disclaimers": "mean",
            "confidence_score": "mean",
        }).round(2)

        for cat in ["factual", "opinion", "commonsense"]:
            if cat in cat_stats.index:
                row = cat_stats.loc[cat]
                print(f"    {cat:<15} avg_words={row['answer_length_words']:.0f}  "
                      f"disclaimers={row['num_disclaimers']:.2f}  "
                      f"confidence={row['confidence_score']:.3f}")

    # Cross-lingual divergence report
    print(f"\n{'='*60}")
    print(f"  Cross-Lingual Divergence Analysis")
    print(f"{'='*60}\n")

    for model_key in df["model"].unique():
        model_df = df[df["model"] == model_key]
        model_name = model_df["model_name"].iloc[0]
        print(f"  🤖 {model_name}")

        # Find questions where response length varies most across languages
        divergence = []
        for qid in model_df["question_id"].unique():
            q_df = model_df[model_df["question_id"] == qid]
            if len(q_df) < 2:
                continue
            length_std = q_df["answer_length_words"].std()
            conf_std = q_df["confidence_score"].std()
            disc_std = q_df["num_disclaimers"].std()
            divergence.append({
                "question_id": qid,
                "category": q_df["category"].iloc[0],
                "length_divergence": round(length_std, 2),
                "confidence_divergence": round(conf_std, 3),
                "disclaimer_divergence": round(disc_std, 2),
                "combined_score": round(length_std * 0.3 + conf_std * 100 * 0.3
                                       + disc_std * 10 * 0.4, 2),
            })

        if divergence:
            div_df = pd.DataFrame(divergence).sort_values(
                "combined_score", ascending=False
            )
            print(f"\n  Top 10 most divergent questions:")
            for i, row in div_df.head(10).iterrows():
                print(f"    Q{row['question_id']:>2d} [{row['category']:<11s}] "
                      f"combined_score={row['combined_score']:.2f}")

    print(f"\n✅ Analysis complete!\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM responses")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model keys to analyze (default: all available)",
    )
    args = parser.parse_args()
    run_analysis(args.models)


if __name__ == "__main__":
    main()
