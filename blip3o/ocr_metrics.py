"""
OCR Metrics — Lumina-level evaluation metrics for OCR tasks.

Ported from Lumina-DiMOO/train/train_unified.py:160-231.
Provides: WER, CER, METEOR, BLEU-2, Edit Distance, Word-level P/R/F1.

Dependencies: evaluate, nltk
"""

import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu


def calculate_metrics(predictions: list, references: list, loaded_metrics=None) -> dict:
    """Lumina-level OCR metrics.

    All comparisons are case-insensitive (.lower()).

    Args:
        predictions: List of predicted strings.
        references:  List of ground-truth strings.
        loaded_metrics: Optional dict with pre-loaded evaluate metrics
                        (keys: "wer", "cer", "meteor") to avoid reloading.

    Returns:
        dict with keys: wer, cer, meteor, bleu, edit_distance, precision, recall, f1
    """
    metrics = {}

    wer_metric = loaded_metrics.get("wer") if loaded_metrics else evaluate.load("wer")
    cer_metric = loaded_metrics.get("cer") if loaded_metrics else evaluate.load("cer")
    meteor_metric = loaded_metrics.get("meteor") if loaded_metrics else evaluate.load("meteor")

    preds_norm = [p.lower() for p in predictions]
    refs_norm = [r.lower() for r in references]

    valid_indices = [i for i, r in enumerate(refs_norm) if len(r.strip()) > 0]
    if not valid_indices:
        return {"wer": 1.0, "cer": 1.0, "meteor": 0.0, "bleu": 0.0,
                "edit_distance": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    preds_norm = [preds_norm[i] for i in valid_indices]
    refs_norm = [refs_norm[i] for i in valid_indices]

    try:
        metrics["wer"] = wer_metric.compute(predictions=preds_norm, references=refs_norm)
    except Exception:
        metrics["wer"] = 1.0

    try:
        metrics["cer"] = cer_metric.compute(predictions=preds_norm, references=refs_norm)
    except Exception:
        metrics["cer"] = 1.0

    try:
        metrics["meteor"] = meteor_metric.compute(predictions=preds_norm, references=refs_norm)["meteor"]
    except Exception:
        metrics["meteor"] = 0.0

    try:
        bleu_scores = []
        for p, r in zip(preds_norm, refs_norm):
            bleu_scores.append(sentence_bleu([r.split()], p.split(), weights=(0.5, 0.5)))
        metrics["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    except Exception:
        metrics["bleu"] = 0.0

    try:
        edit_dists = []
        for p, r in zip(preds_norm, refs_norm):
            edit_dists.append(nltk.edit_distance(p, r))
        metrics["edit_distance"] = sum(edit_dists) / len(edit_dists) if edit_dists else 0.0
    except Exception:
        metrics["edit_distance"] = 0.0

    # Word-level Precision, Recall, F1
    try:
        total_p, total_r, total_f1 = 0.0, 0.0, 0.0
        for p, r in zip(preds_norm, refs_norm):
            p_words = set(p.split())
            r_words = set(r.split())
            if not p_words or not r_words:
                continue
            intersection = p_words.intersection(r_words)
            prec = len(intersection) / len(p_words) if p_words else 0.0
            rec = len(intersection) / len(r_words) if r_words else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            total_p += prec
            total_r += rec
            total_f1 += f1
        metrics["precision"] = total_p / len(preds_norm)
        metrics["recall"] = total_r / len(preds_norm)
        metrics["f1"] = total_f1 / len(preds_norm)
    except Exception:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0

    return metrics
