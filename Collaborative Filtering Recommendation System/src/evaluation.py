import numpy as np
import pandas as pd
from random import sample

def precision_recall_at_k(predictions, k=5, threshold=4.0):

    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        user_est_true.setdefault(uid, []).append((est, true_r))
    
    precisions = []
    recalls = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)  # Sort by estimated rating
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)  # Relevant items
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])  # Recommended items
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                            for est, true_r in user_ratings[:k])  # Hits
        
        precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)
        recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)
    
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    return avg_precision, avg_recall

def evaluate_precision_recall(predictions, k=5, threshold=4.0):

    return precision_recall_at_k(predictions, k=k, threshold=threshold)