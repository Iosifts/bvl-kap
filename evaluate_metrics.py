"""
Comprehensive evaluation of BVL description conditions.

Computes structural, linguistic, and statistical metrics from GPT-classified
clause data, comparing three experimental conditions against expert reference.

Usage:
    python evaluate_metrics.py \\
        --classified-a results/classified_a.json \\
        --classified-b results/classified_b.json \\
        --classified-c results/classified_c.json \\
        --output results/analysis.json

Optional:
    --expert files/bvl_fewshot_gpt_classifier.json  (default)
    --bertscore          enable BERTScore (requires bert_score package)
    --matching-file MAP  JSON mapping expert painting_id -> generated filename

Metrics computed:
  1. Clause distribution profiles (per condition + expert baseline)
  2. Profile similarity to expert (JS divergence, cosine similarity)
  3. Visible-fact ratio and interpretation ratio
  4. Information coverage (class coverage, clause count)
  5. Linguistic quality (sentence length, TTR, Flesch-Kincaid)
  6. Pairwise statistical significance (Wilcoxon signed-rank, Cohen's d)
  7. BERTScore class-conditional (optional, for expert-overlap paintings)
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon, cosine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXPERT = os.path.join(SCRIPT_DIR, "files", "bvl_fewshot_gpt_classifier.json")

CLASSES = ["Overview", "Spatial", "Objects", "Visual", "VisibleText", "Interpretation"]
CONDITION_LABELS = {"a": "BVL only", "b": "BVL + metadata", "c": "BVL + metadata + few-shot"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_classified(path):
    """Load a classified JSON file (output of classify_captions.py)."""
    with open(path) as f:
        return json.load(f)


def load_expert(path):
    """Load expert annotations and compute reference profiles."""
    with open(path) as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Clause profile helpers
# ---------------------------------------------------------------------------

def clause_distribution(clauses):
    """Return normalized distribution over CLASSES."""
    counts = Counter(c.get("class", "Interpretation") for c in clauses)
    total = sum(counts[c] for c in CLASSES)
    if total == 0:
        return np.zeros(len(CLASSES))
    return np.array([counts[c] / total for c in CLASSES])


def clause_counts(clauses):
    """Return raw counts over CLASSES."""
    counts = Counter(c.get("class", "Interpretation") for c in clauses)
    return np.array([counts[c] for c in CLASSES])


def visible_fact_ratio(clauses):
    if not clauses:
        return 0.0
    return sum(1 for c in clauses if c.get("visible_fact", False)) / len(clauses)


def class_coverage(clauses):
    """Number of distinct clause classes present."""
    return len(set(c.get("class") for c in clauses) & set(CLASSES))


# ---------------------------------------------------------------------------
# Expert reference profile
# ---------------------------------------------------------------------------

def compute_expert_profile(expert_data, factual_only=False):
    """Aggregate expert clause distribution across all 10 paintings.

    If factual_only=True, exclude Interpretation clauses to get the
    'ideal factual BVL profile' — the target for generated descriptions
    that are instructed not to interpret.
    """
    all_clauses = []
    for painting in expert_data:
        all_clauses.extend(painting["output"])
    if factual_only:
        all_clauses = [c for c in all_clauses if c.get("visible_fact", True)]
    return clause_distribution(all_clauses)


def compute_expert_per_painting(expert_data):
    """Per-painting expert distributions, keyed by painting_id."""
    profiles = {}
    for painting in expert_data:
        pid = painting["painting_id"]
        factual_clauses = [c for c in painting["output"]
                           if c.get("visible_fact", True)]
        profiles[pid] = {
            "distribution": clause_distribution(painting["output"]),
            "distribution_factual": clause_distribution(factual_clauses),
            "visible_fact_ratio": visible_fact_ratio(painting["output"]),
            "class_coverage": class_coverage(painting["output"]),
            "clause_count": len(painting["output"]),
            "clauses": painting["output"],
        }
    return profiles


# ---------------------------------------------------------------------------
# Linguistic metrics
# ---------------------------------------------------------------------------

def tokenize_words(text):
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())


def syllable_count(word):
    """Rough syllable count for Flesch-Kincaid."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e"):
        count -= 1
    return max(count, 1)


def flesch_kincaid_grade(text):
    """Compute Flesch-Kincaid Grade Level."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = tokenize_words(text)
    if not sentences or not words:
        return 0.0
    avg_sentence_len = len(words) / len(sentences)
    avg_syllables = sum(syllable_count(w) for w in words) / len(words)
    return 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59


def type_token_ratio(text):
    """Vocabulary diversity: unique words / total words."""
    words = tokenize_words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_linguistic_metrics(caption_text):
    """Compute linguistic metrics for a single description."""
    words = tokenize_words(caption_text)
    sentences = re.split(r'[.!?]+', caption_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "type_token_ratio": type_token_ratio(caption_text),
        "flesch_kincaid_grade": flesch_kincaid_grade(caption_text),
    }


# ---------------------------------------------------------------------------
# Profile similarity
# ---------------------------------------------------------------------------

def js_divergence(p, q):
    """Jensen-Shannon divergence (0 = identical, 1 = maximally different)."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    # add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q) ** 2)  # squared = actual JSD


def cosine_similarity(p, q):
    """Cosine similarity (1 = identical, 0 = orthogonal)."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    if np.all(p == 0) or np.all(q == 0):
        return 0.0
    return float(1 - cosine(p, q))


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_wilcoxon(values_a, values_b):
    """Wilcoxon signed-rank test for paired samples."""
    diffs = np.array(values_a) - np.array(values_b)
    diffs = diffs[diffs != 0]
    if len(diffs) < 10:
        return {"statistic": None, "p_value": None, "n_nonzero": len(diffs),
                "note": "too few non-zero differences"}
    stat, p = stats.wilcoxon(values_a, values_b)
    return {"statistic": float(stat), "p_value": float(p),
            "n_nonzero": int(len(diffs))}


def cohens_d(values_a, values_b):
    """Cohen's d effect size for paired samples."""
    diffs = np.array(values_a) - np.array(values_b)
    if np.std(diffs) == 0:
        return 0.0
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def run_pairwise_tests(metric_per_painting, conditions):
    """Run Wilcoxon + Cohen's d for all condition pairs."""
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    results = {}
    for c1, c2 in pairs:
        if c1 not in metric_per_painting or c2 not in metric_per_painting:
            continue
        v1 = metric_per_painting[c1]
        v2 = metric_per_painting[c2]
        # align by file (assume same order)
        n = min(len(v1), len(v2))
        results[f"{c1}_vs_{c2}"] = {
            "wilcoxon": paired_wilcoxon(v1[:n], v2[:n]),
            "cohens_d": cohens_d(v1[:n], v2[:n]),
            "n": n,
        }
    return results


# ---------------------------------------------------------------------------
# BERTScore (optional)
# ---------------------------------------------------------------------------

def compute_bertscore_optional(expert_data, classified_conditions, matching):
    """Compute BERTScore between generated and expert text.

    Returns per-class and overall BERTScore for overlapping paintings.
    Requires: pip install bert_score
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return {"error": "bert_score not installed. pip install bert_score"}

    expert_by_id = {p["painting_id"]: p for p in expert_data}
    results = {}

    for cond_key, entries in classified_conditions.items():
        cond_results = {}
        for entry in entries:
            fname = entry["file"]
            pid = matching.get(fname)
            if pid is None or pid not in expert_by_id:
                continue

            expert = expert_by_id[pid]
            # overall BERTScore
            gen_text = entry.get("caption", "")
            ref_text = expert["input_description"]
            P, R, F = bert_score_fn(
                [gen_text], [ref_text], lang="en", verbose=False)
            overall = {"precision": P[0].item(), "recall": R[0].item(),
                       "f1": F[0].item()}

            # class-conditional BERTScore
            per_class = {}
            expert_by_class = {}
            for cl in expert["output"]:
                expert_by_class.setdefault(cl["class"], []).append(cl["text"])
            gen_by_class = {}
            for cl in entry.get("clauses", []):
                gen_by_class.setdefault(cl["class"], []).append(cl["text"])

            for cls in CLASSES:
                gen_texts = gen_by_class.get(cls, [])
                ref_texts = expert_by_class.get(cls, [])
                if not gen_texts or not ref_texts:
                    per_class[cls] = None
                    continue
                # compare each generated clause against concatenated expert
                ref_combined = " ".join(ref_texts)
                P, R, F = bert_score_fn(
                    gen_texts, [ref_combined] * len(gen_texts),
                    lang="en", verbose=False)
                per_class[cls] = {
                    "precision": float(P.mean()),
                    "recall": float(R.mean()),
                    "f1": float(F.mean()),
                }

            cond_results[pid] = {"overall": overall, "per_class": per_class}
        results[cond_key] = cond_results

    return results


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_condition(entries):
    """Compute all per-entry metrics for one condition."""
    per_entry = []
    for entry in entries:
        clauses = entry.get("clauses", [])
        caption = entry.get("caption", "")
        dist = clause_distribution(clauses)
        ling = compute_linguistic_metrics(caption)
        per_entry.append({
            "file": entry["file"],
            "distribution": dist.tolist(),
            "clause_count": len(clauses),
            "class_coverage": class_coverage(clauses),
            "visible_fact_ratio": visible_fact_ratio(clauses),
            "interpretation_ratio": (
                sum(1 for c in clauses if c.get("class") == "Interpretation")
                / max(len(clauses), 1)
            ),
            "linguistic": ling,
        })
    return per_entry


def aggregate_condition(per_entry):
    """Compute aggregate statistics for a condition."""
    if not per_entry:
        return {}
    n = len(per_entry)

    def mean_field(field):
        vals = [e[field] for e in per_entry]
        return float(np.mean(vals))

    def std_field(field):
        vals = [e[field] for e in per_entry]
        return float(np.std(vals, ddof=1)) if n > 1 else 0.0

    def mean_ling(field):
        vals = [e["linguistic"][field] for e in per_entry]
        return float(np.mean(vals))

    def std_ling(field):
        vals = [e["linguistic"][field] for e in per_entry]
        return float(np.std(vals, ddof=1)) if n > 1 else 0.0

    avg_dist = np.mean([e["distribution"] for e in per_entry], axis=0)

    return {
        "n": n,
        "avg_distribution": avg_dist.tolist(),
        "avg_clause_count": mean_field("clause_count"),
        "std_clause_count": std_field("clause_count"),
        "avg_class_coverage": mean_field("class_coverage"),
        "std_class_coverage": std_field("class_coverage"),
        "avg_visible_fact_ratio": mean_field("visible_fact_ratio"),
        "std_visible_fact_ratio": std_field("visible_fact_ratio"),
        "avg_interpretation_ratio": mean_field("interpretation_ratio"),
        "std_interpretation_ratio": std_field("interpretation_ratio"),
        "avg_word_count": mean_ling("word_count"),
        "std_word_count": std_ling("word_count"),
        "avg_sentence_count": mean_ling("sentence_count"),
        "std_sentence_count": std_ling("sentence_count"),
        "avg_sentence_length": mean_ling("avg_sentence_length"),
        "std_sentence_length": std_ling("avg_sentence_length"),
        "avg_ttr": mean_ling("type_token_ratio"),
        "std_ttr": std_ling("type_token_ratio"),
        "avg_flesch_kincaid": mean_ling("flesch_kincaid_grade"),
        "std_flesch_kincaid": std_ling("flesch_kincaid_grade"),
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_table_1(aggs, expert_dist):
    """Table 1: Clause Distribution Profiles (%)."""
    print("\n" + "=" * 80)
    print("TABLE 1: Clause Distribution Profiles (%)")
    print("=" * 80)
    header = f"{'Class':<16}"
    for label in ["Expert"] + [CONDITION_LABELS[k] for k in sorted(aggs)]:
        header += f"{label:>18}"
    print(header)
    print("-" * 80)

    dists = {"expert": expert_dist}
    for k in sorted(aggs):
        dists[k] = aggs[k]["avg_distribution"]

    for i, cls in enumerate(CLASSES):
        row = f"{cls:<16}"
        row += f"{expert_dist[i]*100:>17.1f}%"
        for k in sorted(aggs):
            row += f"{aggs[k]['avg_distribution'][i]*100:>17.1f}%"
        print(row)
    print()


def print_table_2(aggs, expert_dist):
    """Table 2: Profile Similarity to Expert."""
    print("=" * 80)
    print("TABLE 2: Profile Similarity to Expert Reference")
    print("=" * 80)
    print(f"{'Condition':<28} {'JS Divergence':>14} {'Cosine Sim':>14}")
    print("-" * 60)
    for k in sorted(aggs):
        dist = aggs[k]["avg_distribution"]
        jsd = js_divergence(dist, expert_dist)
        cos = cosine_similarity(dist, expert_dist)
        print(f"{CONDITION_LABELS[k]:<28} {jsd:>14.4f} {cos:>14.4f}")
    print()
    print("  JS Divergence: 0 = identical, 1 = maximally different")
    print("  Cosine Similarity: 1 = identical direction, 0 = orthogonal")
    print()


def print_table_3(aggs, expert_data):
    """Table 3: Factuality and Coverage."""
    # compute expert baselines
    all_expert_clauses = []
    expert_coverages = []
    for p in expert_data:
        all_expert_clauses.extend(p["output"])
        expert_coverages.append(class_coverage(p["output"]))
    expert_vfr = visible_fact_ratio(all_expert_clauses)
    expert_ir = sum(1 for c in all_expert_clauses if c.get("class") == "Interpretation") / max(len(all_expert_clauses), 1)
    expert_cc = np.mean(expert_coverages)

    print("=" * 80)
    print("TABLE 3: Factuality and Information Coverage")
    print("=" * 80)
    header = f"{'Metric':<28} {'Expert':>12}"
    for k in sorted(aggs):
        header += f"  {CONDITION_LABELS[k]:>18}"
    print(header)
    print("-" * 80)

    metrics = [
        ("Visible-fact ratio", "avg_visible_fact_ratio", expert_vfr),
        ("Interpretation ratio", "avg_interpretation_ratio", expert_ir),
        ("Class coverage (of 6)", "avg_class_coverage", expert_cc),
        ("Clauses per description", "avg_clause_count", np.mean([len(p["output"]) for p in expert_data])),
    ]
    for label, key, expert_val in metrics:
        row = f"{label:<28} {expert_val:>12.3f}"
        for k in sorted(aggs):
            row += f"  {aggs[k][key]:>18.3f}"
        print(row)
    print()


def print_table_4(aggs):
    """Table 4: Linguistic Quality."""
    print("=" * 80)
    print("TABLE 4: Linguistic Quality Metrics")
    print("=" * 80)
    header = f"{'Metric':<28}"
    for k in sorted(aggs):
        header += f"  {CONDITION_LABELS[k]:>18}"
    print(header)
    print("-" * 80)

    metrics = [
        ("Word count", "avg_word_count", "std_word_count"),
        ("Sentence count", "avg_sentence_count", "std_sentence_count"),
        ("Avg sentence length", "avg_sentence_length", "std_sentence_length"),
        ("Type-token ratio", "avg_ttr", "std_ttr"),
        ("Flesch-Kincaid grade", "avg_flesch_kincaid", "std_flesch_kincaid"),
    ]
    for label, mean_key, std_key in metrics:
        row = f"{label:<28}"
        for k in sorted(aggs):
            m = aggs[k][mean_key]
            s = aggs[k][std_key]
            row += f"  {m:>10.2f} ±{s:<6.2f}"
        print(row)
    print()


def print_table_5(stat_results):
    """Table 5: Statistical Significance."""
    print("=" * 80)
    print("TABLE 5: Pairwise Statistical Tests")
    print("=" * 80)

    for metric_name, pairs in stat_results.items():
        print(f"\n  {metric_name}:")
        print(f"    {'Comparison':<24} {'Cohen d':>10} {'p-value':>12} {'Sig':>6}")
        print("    " + "-" * 56)
        for pair_name, result in pairs.items():
            d = result["cohens_d"]
            w = result["wilcoxon"]
            p_str = f"{w['p_value']:.4f}" if w["p_value"] is not None else "n/a"
            sig = ""
            if w["p_value"] is not None:
                if w["p_value"] < 0.001:
                    sig = "***"
                elif w["p_value"] < 0.01:
                    sig = "**"
                elif w["p_value"] < 0.05:
                    sig = "*"
            label = pair_name.replace("_vs_", " vs ")
            label = " vs ".join(CONDITION_LABELS.get(x, x) for x in pair_name.split("_vs_"))
            print(f"    {label:<24} {d:>10.3f} {p_str:>12} {sig:>6}")
    print()
    print("  * p < .05, ** p < .01, *** p < .001")
    print("  Cohen's d: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive BVL description evaluation metrics")
    parser.add_argument("--classified-a", required=True,
                        help="Classified JSON for condition A (BVL only)")
    parser.add_argument("--classified-b", required=True,
                        help="Classified JSON for condition B (BVL + metadata)")
    parser.add_argument("--classified-c", required=True,
                        help="Classified JSON for condition C (BVL + metadata + few-shot)")
    parser.add_argument("--expert", default=DEFAULT_EXPERT,
                        help="Expert annotations JSON")
    parser.add_argument("--output", default="analysis_results.json",
                        help="Output JSON file")
    parser.add_argument("--bertscore", action="store_true",
                        help="Compute BERTScore (requires bert_score package)")
    parser.add_argument("--matching-file",
                        help="JSON mapping generated filenames to expert painting_id")
    args = parser.parse_args()

    # load data
    classified = {
        "a": load_classified(args.classified_a),
        "b": load_classified(args.classified_b),
        "c": load_classified(args.classified_c),
    }
    expert_data = load_expert(args.expert)
    expert_dist_full = compute_expert_profile(expert_data, factual_only=False)
    expert_dist_factual = compute_expert_profile(expert_data, factual_only=True)

    print(f"Loaded: A={len(classified['a'])}, B={len(classified['b'])}, "
          f"C={len(classified['c'])} entries, Expert={len(expert_data)} paintings")

    # per-entry analysis
    per_entry = {}
    aggs = {}
    for cond_key, entries in classified.items():
        per_entry[cond_key] = analyze_condition(entries)
        aggs[cond_key] = aggregate_condition(per_entry[cond_key])

    # ---------- Tables ----------
    print("\n>>> Using FACTUAL-ONLY expert profile (Interpretation clauses removed)")
    print(">>> This is the correct reference for generated descriptions that")
    print(">>> are instructed not to interpret.\n")
    print_table_1(aggs, expert_dist_factual)
    print_table_2(aggs, expert_dist_factual)
    print_table_3(aggs, expert_data)
    print_table_4(aggs)

    print("\n>>> For reference: FULL expert profile (including Interpretation)")
    print_table_2(aggs, expert_dist_full)

    # ---------- Statistical tests ----------
    metric_vectors = {}
    for metric_name, field in [
        ("visible_fact_ratio", "visible_fact_ratio"),
        ("interpretation_ratio", "interpretation_ratio"),
        ("class_coverage", "class_coverage"),
        ("clause_count", "clause_count"),
    ]:
        metric_vectors[metric_name] = {
            k: [e[field] for e in per_entry[k]] for k in per_entry
        }
    for metric_name, field in [
        ("word_count", "word_count"),
        ("avg_sentence_length", "avg_sentence_length"),
        ("type_token_ratio", "type_token_ratio"),
        ("flesch_kincaid_grade", "flesch_kincaid_grade"),
    ]:
        metric_vectors[metric_name] = {
            k: [e["linguistic"][field] for e in per_entry[k]] for k in per_entry
        }
    # JS divergence per painting to expert (only for per-class distribution comparison)
    # We'll also test the per-painting JS divergence to expert as a paired metric
    # between conditions (which condition's paintings are closer to expert?)
    # This requires per-painting distributions, which we already have.

    stat_results = {}
    for metric_name, vectors in metric_vectors.items():
        stat_results[metric_name] = run_pairwise_tests(vectors, classified.keys())

    print_table_5(stat_results)

    # ---------- BERTScore (optional) ----------
    bertscore_results = None
    if args.bertscore:
        matching = {}
        if args.matching_file:
            with open(args.matching_file) as f:
                matching = json.load(f)
        if not matching:
            print("WARNING: --bertscore requires --matching-file to map "
                  "generated filenames to expert painting_ids. Skipping.")
        else:
            print("Computing BERTScore (this may take a while)...")
            bertscore_results = compute_bertscore_optional(
                expert_data, classified, matching)

    # ---------- Save results ----------
    output = {
        "conditions": {},
        "expert_profile_full": {
            "distribution": expert_dist_full.tolist(),
            "classes": CLASSES,
            "note": "includes Interpretation clauses",
        },
        "expert_profile_factual": {
            "distribution": expert_dist_factual.tolist(),
            "classes": CLASSES,
            "note": "visible_fact=true only, Interpretation removed",
        },
        "profile_similarity_factual": {},
        "profile_similarity_full": {},
        "statistical_tests": {},
    }

    for k in sorted(aggs):
        output["conditions"][CONDITION_LABELS[k]] = {
            "aggregate": aggs[k],
            "per_entry_count": len(per_entry[k]),
        }
        # factual-only comparison (primary)
        jsd_f = js_divergence(aggs[k]["avg_distribution"], expert_dist_factual)
        cos_f = cosine_similarity(aggs[k]["avg_distribution"], expert_dist_factual)
        output["profile_similarity_factual"][CONDITION_LABELS[k]] = {
            "js_divergence": jsd_f,
            "cosine_similarity": cos_f,
        }
        # full comparison (for reference)
        jsd = js_divergence(aggs[k]["avg_distribution"], expert_dist_full)
        cos = cosine_similarity(aggs[k]["avg_distribution"], expert_dist_full)
        output["profile_similarity_full"][CONDITION_LABELS[k]] = {
            "js_divergence": jsd,
            "cosine_similarity": cos,
        }

    # serialize stat results (convert numpy types)
    for metric_name, pairs in stat_results.items():
        output["statistical_tests"][metric_name] = {}
        for pair_name, result in pairs.items():
            output["statistical_tests"][metric_name][pair_name] = {
                "cohens_d": result["cohens_d"],
                "wilcoxon_p": result["wilcoxon"]["p_value"],
                "wilcoxon_stat": result["wilcoxon"]["statistic"],
                "n": result["n"],
            }

    if bertscore_results:
        output["bertscore"] = bertscore_results

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
