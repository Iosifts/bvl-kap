"""
BVL Clause Classifier — Train on GPT-labeled data, test on expert annotations.

Trains a RoBERTa-base classifier on clause-level BVL categories.
Four experiments: one per condition (A, B, C) and one combined (A+B+C).
Each is evaluated on a held-out validation set and on the expert test set.

Usage:
    python train_classifier.py [--epochs 10] [--batch-size 32] [--lr 2e-5] [--seed 42]

Outputs (in models/ directory):
    models/condition_a/          — best model for condition A
    models/condition_b/          — best model for condition B
    models/condition_c/          — best model for condition C
    models/combined/             — best model trained on all conditions
    models/training_log.json     — full training log with all metrics
    models/final_results.json    — summary comparison table

Requires: torch, transformers, scikit-learn
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL2ID = {
    "Overview": 0,
    "Spatial": 1,
    "Objects": 2,
    "Visual": 3,
    "VisibleText": 4,
    "Interpretation": 5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

MODEL_NAME = "roberta-base"

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
MODELS_DIR = SCRIPT_DIR / "models"
EXPERT_FILE = SCRIPT_DIR / "files" / "bvl_fewshot_gpt_classifier.json"

CLASSIFIED_FILES = {
    "condition_a": RESULTS_DIR / "classified_a.json",
    "condition_b": RESULTS_DIR / "classified_b.json",
    "condition_c": RESULTS_DIR / "classified_c.json",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bvl-classifier")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_classified_clauses(path):
    """Load clauses from a classified JSON file. Returns list of (text, label)."""
    with open(path) as f:
        data = json.load(f)
    clauses = []
    for entry in data:
        for clause in entry.get("clauses", []):
            text = clause.get("text", "").strip()
            label = clause.get("class", "")
            if text and label in LABEL2ID:
                clauses.append((text, LABEL2ID[label]))
    return clauses


def load_expert_clauses():
    """Load expert-annotated clauses as test set."""
    with open(EXPERT_FILE) as f:
        data = json.load(f)
    clauses = []
    for entry in data:
        for clause in entry.get("output", []):
            text = clause.get("text", "").strip()
            label = clause.get("class", "")
            if text and label in LABEL2ID:
                clauses.append((text, LABEL2ID[label]))
    return clauses


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ClauseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_sampler(labels):
    """Weighted random sampler to handle class imbalance."""
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        total_loss += loss.item() * labels.size(0)

        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item() * labels.size(0)
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        target_names=[ID2LABEL[i] for i in range(NUM_LABELS)],
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_LABELS)))

    return {
        "loss": total_loss / n,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "uar": uar,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
    }


def run_experiment(name, train_clauses, val_clauses, test_clauses, args, device):
    """Full train→validate→test pipeline for one experiment."""
    log.info("=" * 60)
    log.info(f"EXPERIMENT: {name}")
    log.info(f"  Train: {len(train_clauses)}  Val: {len(val_clauses)}  "
             f"Test (expert): {len(test_clauses)}")

    train_dist = Counter(l for _, l in train_clauses)
    log.info(f"  Train distribution: "
             + ", ".join(f"{ID2LABEL[k]}={v}" for k, v in sorted(train_dist.items())))

    # Tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_texts, train_labels = zip(*train_clauses)
    val_texts, val_labels = zip(*val_clauses)
    test_texts, test_labels = zip(*test_clauses)

    train_ds = ClauseDataset(list(train_texts), list(train_labels), tokenizer)
    val_ds = ClauseDataset(list(val_texts), list(val_labels), tokenizer)
    test_ds = ClauseDataset(list(test_texts), list(test_labels), tokenizer)

    sampler = make_sampler(list(train_labels))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, num_workers=0)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    save_dir = MODELS_DIR / name
    epoch_logs = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
        )
        val_result = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        log.info(
            f"  Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_result['loss']:.4f} val_acc={val_result['accuracy']:.3f} "
            f"val_F1={val_result['macro_f1']:.3f} val_UAR={val_result['uar']:.3f} | "
            f"{elapsed:.1f}s"
        )

        epoch_log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_result["loss"], 5),
            "val_acc": round(val_result["accuracy"], 4),
            "val_macro_f1": round(val_result["macro_f1"], 4),
            "val_weighted_f1": round(val_result["weighted_f1"], 4),
            "val_uar": round(val_result["uar"], 4),
        }
        epoch_logs.append(epoch_log)

        if val_result["macro_f1"] > best_val_f1:
            best_val_f1 = val_result["macro_f1"]
            best_epoch = epoch
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            log.info(f"    ↑ New best val macro-F1={best_val_f1:.4f}, saved to {save_dir}")

    log.info(f"  Best epoch: {best_epoch} (val macro-F1={best_val_f1:.4f})")

    # Reload best model for final test
    log.info(f"  Loading best model from {save_dir} for expert test...")
    model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
    test_result = evaluate(model, test_loader, device)

    log.info(f"  EXPERT TEST — acc={test_result['accuracy']:.3f} "
             f"macro-F1={test_result['macro_f1']:.3f} "
             f"weighted-F1={test_result['weighted_f1']:.3f} "
             f"UAR={test_result['uar']:.3f}")

    # Per-class F1 on expert test
    for cls_name in LABEL2ID:
        cls_metrics = test_result["report"].get(cls_name, {})
        f1 = cls_metrics.get("f1-score", 0)
        sup = cls_metrics.get("support", 0)
        log.info(f"    {cls_name:15s} F1={f1:.3f}  support={sup}")

    return {
        "name": name,
        "train_size": len(train_clauses),
        "val_size": len(val_clauses),
        "test_size": len(test_clauses),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_accuracy": round(test_result["accuracy"], 4),
        "test_macro_f1": round(test_result["macro_f1"], 4),
        "test_weighted_f1": round(test_result["weighted_f1"], 4),
        "test_uar": round(test_result["uar"], 4),
        "test_report": test_result["report"],
        "test_confusion_matrix": test_result["confusion_matrix"],
        "epoch_logs": epoch_logs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="BVL Clause Classifier Training")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    parser.add_argument(
        "--conditions", nargs="*", default=None,
        help="Which conditions to run: a b c combined (default: all). "
             "E.g. --conditions a combined",
    )
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        log.info("  GPU: Apple Silicon (MPS)")

    # Load expert test set (shared across all experiments)
    test_clauses = load_expert_clauses()
    test_dist = Counter(l for _, l in test_clauses)
    log.info(f"Expert test set: {len(test_clauses)} clauses")
    log.info("  " + ", ".join(f"{ID2LABEL[k]}={v}" for k, v in sorted(test_dist.items())))

    # Load generated data per condition
    condition_data = {}
    for cond_name, path in CLASSIFIED_FILES.items():
        if not path.exists():
            log.warning(f"  {path} not found, skipping {cond_name}")
            continue
        clauses = load_classified_clauses(path)
        condition_data[cond_name] = clauses
        log.info(f"Loaded {cond_name}: {len(clauses)} clauses")

    if not condition_data:
        log.error("No training data found. Exiting.")
        sys.exit(1)

    # Decide which experiments to run
    all_experiments = list(condition_data.keys()) + ["combined"]
    if args.conditions:
        requested = []
        for c in args.conditions:
            key = f"condition_{c}" if not c.startswith("condition_") and c != "combined" else c
            requested.append(key)
        experiments = [e for e in requested if e in all_experiments]
    else:
        experiments = all_experiments

    log.info(f"Experiments to run: {experiments}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for exp_name in experiments:
        if exp_name == "combined":
            all_clauses = []
            for clauses in condition_data.values():
                all_clauses.extend(clauses)
        else:
            all_clauses = condition_data[exp_name]

        texts, labels = zip(*all_clauses)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=args.val_ratio,
            stratify=labels, random_state=args.seed,
        )
        train_clauses = list(zip(train_texts, train_labels))
        val_clauses = list(zip(val_texts, val_labels))

        result = run_experiment(
            exp_name, train_clauses, val_clauses, test_clauses, args, device,
        )
        all_results[exp_name] = result

    # Save full training log
    log_path = MODELS_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Full training log saved to {log_path}")

    # Save summary comparison table
    summary = []
    for name, r in all_results.items():
        summary.append({
            "experiment": name,
            "train_size": r["train_size"],
            "best_val_F1": r["best_val_macro_f1"],
            "expert_accuracy": r["test_accuracy"],
            "expert_macro_F1": r["test_macro_f1"],
            "expert_weighted_F1": r["test_weighted_f1"],
            "expert_UAR": r["test_uar"],
        })
    summary_path = MODELS_DIR / "final_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    log.info("")
    log.info("=" * 70)
    log.info("FINAL COMPARISON")
    log.info("=" * 70)
    log.info(f"{'Experiment':<18} {'Train':>6} {'Val F1':>7} {'Test Acc':>9} "
             f"{'Test mF1':>9} {'Test wF1':>9} {'Test UAR':>9}")
    log.info("-" * 80)
    for s in summary:
        log.info(f"{s['experiment']:<18} {s['train_size']:>6} "
                 f"{s['best_val_F1']:>7.4f} {s['expert_accuracy']:>9.4f} "
                 f"{s['expert_macro_F1']:>9.4f} {s['expert_weighted_F1']:>9.4f} "
                 f"{s['expert_UAR']:>9.4f}")
    log.info("=" * 70)
    log.info(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
