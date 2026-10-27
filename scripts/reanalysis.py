"""
Re-check of the headline numbers in cross-lingual-llm-bias.

Usage (from the repository root):
    python scripts/reanalysis.py .

Prints three checks:
  1. Hedging in opinion answers, Russian vs English: original (asymmetric) patterns
     vs the mirrored patterns now used in scripts/analyze_responses.py.
  2. Jais 30B Kazakh replies: how many repeat the question / contain the same phrase.
  3. Kazakh vs English answer length, adjusted for how long each question is in each language.

Only the standard library is needed (exact binomial tests are computed directly).
"""

import ast
import json
import re
import sys
from math import comb
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Kazakh letters in the output on Windows consoles
repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def read_patterns(path):
    """Read DISCLAIMER_PATTERNS from analyze_responses.py without importing it (no pandas needed)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DISCLAIMER_PATTERNS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"DISCLAIMER_PATTERNS not found in {path}")


MODELS = ["llama3-8b", "llama3-70b", "jais-30b"]

# The patterns the first version of the report was based on. They are not symmetric:
# Russian counts every "однако", English counts "however" only before "it's"/"it is"/"there".
# Kept here only to reproduce the old 12 of 15 vs 4 of 15 result.
ORIGINAL = {
    "en": [r"\bi('m| am) (just )?an? (ai|language model|assistant)\b",
           r"\bi (cannot|can't|don't) (provide|give|offer)\b",
           r"\bit('s| is) (important|worth) (to note|noting|mentioning)\b",
           r"\bthis is (a )?(complex|nuanced|subjective)\b",
           r"\bthere (are|is) no (single|one|definitive) (right )?answer\b",
           r"\bI (should|must) (note|mention|clarify)\b",
           r"\bdisclaimer\b", r"\bin my opinion\b", r"\bsome people (believe|think|argue)\b",
           r"\bit depends on\b", r"\bhowever,? (it('s| is)|there)\b", r"\bon the other hand\b"],
    "ru": [r"\bя (являюсь|—) (искусственным интеллектом|языковой моделью|ии)\b",
           r"\bважно (отметить|заметить|учитывать)\b",
           r"\bэто (сложный|субъективный|неоднозначный) вопрос\b",
           r"\bоднозначного ответа нет\b", r"\bнекоторые (считают|полагают|думают)\b",
           r"\bс одной стороны\b", r"\bс другой стороны\b", r"\bзависит от\b",
           r"\bследует отметить\b", r"\bоднако\b"],
}
MIRRORED = read_patterns(repo / "scripts" / "analyze_responses.py")


def binom_two_sided(k, n):
    """Exact two-sided binomial test against p = 0.5 (same result as scipy.stats.binomtest)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(model):
    path = repo / "results" / "responses" / f"{model}_responses.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["responses"]
    return {(r["question_id"], r["language"]): r for r in rows}


def mcnemar(flags_a, flags_b):
    """Exact McNemar test on paired yes/no flags (same questions in two languages)."""
    only_a = sum(a and not b for a, b in zip(flags_a, flags_b))
    only_b = sum(b and not a for a, b in zip(flags_a, flags_b))
    return only_a, only_b, binom_two_sided(only_a, only_a + only_b)


def has_hedge(patterns, text, lang):
    return any(re.search(p, (text or "").lower()) for p in patterns[lang])


data = {m: load(m) for m in MODELS}
opinion_ids = sorted({q for (q, _), r in data["llama3-8b"].items() if r["category"] == "opinion"})
all_ids = sorted({q for q, _ in data["llama3-8b"]})

print(f"1. Hedging in opinion answers (questions with at least one match, of {len(opinion_ids)})")
for m in MODELS:
    for name, patterns in [("original", ORIGINAL), ("mirrored", MIRRORED)]:
        ru = [has_hedge(patterns, data[m][(q, "ru")]["answer"], "ru") for q in opinion_ids]
        en = [has_hedge(patterns, data[m][(q, "en")]["answer"], "en") for q in opinion_ids]
        only_ru, only_en, p = mcnemar(ru, en)
        print(f"   {m:11s} {name:9s} ru {sum(ru):2d}/{len(ru)}  en {sum(en):2d}/{len(en)}  "
              f"(only ru {only_ru}, only en {only_en}, McNemar p = {p:.3f})")

print("\n2. Jais 30B Kazakh replies")
phrase = "дүниежүзілігі дамуы бар"
kz = [data["jais-30b"][(q, "kz")] for q in all_ids]
echo = sum((r["answer"] or "").strip().startswith(r["question"].strip()[:25]) for r in kz)
same = sum(phrase in (r["answer"] or "") for r in kz)
print(f"   repeat the question: {echo}/{len(kz)}, contain the same phrase ({phrase}...): {same}/{len(kz)}")

print("\n3. Kazakh vs English answer length, in characters, adjusted for question length")
print("   ratio = (kz answer / en answer) / (kz question / en question); < 1 means Kazakh answer is shorter")
for m in MODELS:
    ratios = []
    for q in all_ids:
        a_kz, a_en = data[m][(q, "kz")], data[m][(q, "en")]
        answer_ratio = len(a_kz["answer"] or "") / len(a_en["answer"] or "")
        question_ratio = len(a_kz["question"]) / len(a_en["question"])
        ratios.append(answer_ratio / question_ratio)
    shorter = sum(r < 1 for r in ratios)
    median = sorted(ratios)[len(ratios) // 2]
    p = binom_two_sided(shorter, len(ratios))
    print(f"   {m:11s} Kazakh shorter on {shorter}/{len(ratios)} questions, "
          f"median ratio {median:.2f}, sign test p = {p:.3f}")
