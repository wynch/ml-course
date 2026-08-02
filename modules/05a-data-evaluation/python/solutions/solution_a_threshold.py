"""Solution A — threshold selection."""

from src.metrics import classification_metrics


def best_f1_threshold(y_true, probability, candidates):
    best_threshold = None
    best_score = -1.0
    for threshold in sorted(candidates):
        score = classification_metrics(y_true, probability, threshold)["f1"]
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold
