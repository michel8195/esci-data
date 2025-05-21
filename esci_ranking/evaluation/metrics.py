"""
Evaluation metrics for ranking models
"""

import torch
import numpy as np
from sklearn.metrics import ndcg_score
import logging

logger = logging.getLogger(__name__)


def compute_metrics(similarity_scores, gains, num_products):
    """
    Compute ranking metrics
    
    Args:
        similarity_scores: Tensor of similarity scores
        gains: Tensor of gain values
        num_products: List of number of products per query
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Process each query separately
    start_idx = 0
    ndcg_scores = []
    recall_scores = {1: [], 5: [], 10: []}
    
    for num_prods in num_products:
        # Get scores and gains for this query
        query_scores = similarity_scores[start_idx:start_idx + num_prods]
        query_gains = gains[start_idx:start_idx + num_prods]
        
        # Convert to numpy for sklearn metrics
        query_scores_np = query_scores.cpu().numpy().reshape(1, -1)
        query_gains_np = query_gains.cpu().numpy().reshape(1, -1)
        
        # Compute NDCG@10
        if num_prods >= 10:
            ndcg = ndcg_score(query_gains_np, query_scores_np, k=10)
            ndcg_scores.append(ndcg)
        
        # Compute Recall@k
        # Sort by scores
        _, indices = torch.sort(query_scores, descending=True)
        sorted_gains = query_gains[indices]
        
        # Compute recall at different k values
        for k in recall_scores.keys():
            if num_prods >= k:
                # Count relevant items (gain > 0) in top-k
                relevant_in_topk = (sorted_gains[:k] > 0).sum().item()
                total_relevant = (query_gains > 0).sum().item()
                
                if total_relevant > 0:
                    recall = relevant_in_topk / total_relevant
                    recall_scores[k].append(recall)
        
        start_idx += num_prods
    
    # Average metrics across queries
    if ndcg_scores:
        metrics['ndcg@10'] = np.mean(ndcg_scores)
    
    for k, scores in recall_scores.items():
        if scores:
            metrics[f'recall@{k}'] = np.mean(scores)
    
    return metrics


def compute_ndcg(y_true, y_score, k=10):
    """
    Compute NDCG@k for a single query
    
    Args:
        y_true: Array of true relevance scores
        y_score: Array of predicted scores
        k: Number of results to consider
        
    Returns:
        NDCG@k score
    """
    return ndcg_score(np.array([y_true]), np.array([y_score]), k=k)


def compute_recall_at_k(y_true, y_score, k=10, threshold=0):
    """
    Compute Recall@k for a single query
    
    Args:
        y_true: Array of true relevance scores
        y_score: Array of predicted scores
        k: Number of results to consider
        threshold: Relevance threshold
        
    Returns:
        Recall@k score
    """
    # Sort by predicted scores
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_true = y_true[sorted_indices]
    
    # Count relevant items in top-k
    relevant_in_topk = np.sum(sorted_true[:k] > threshold)
    total_relevant = np.sum(y_true > threshold)
    
    if total_relevant == 0:
        return 0.0
    
    return relevant_in_topk / total_relevant


def evaluate_rankings(rankings_df, ground_truth_df, k_values=[1, 5, 10]):
    """
    Evaluate rankings against ground truth
    
    Args:
        rankings_df: DataFrame with rankings (query_id, product_id, score)
        ground_truth_df: DataFrame with ground truth (query_id, product_id, esci_label)
        k_values: List of k values for evaluation
        
    Returns:
        Dictionary of metrics
    """
    # Convert ESCI labels to gain values
    esci_label2gain = {
        'exact': 1.0,
        'substitute': 0.1,
        'complement': 0.01,
        'irrelevant': 0.0,
    }
    
    ground_truth_df['gain'] = ground_truth_df['esci_label'].map(esci_label2gain)
    
    # Group by query_id
    query_groups = rankings_df.groupby('query_id')
    ground_truth_groups = ground_truth_df.groupby('query_id')
    
    # Metrics to compute
    metrics = {
        'ndcg': {k: [] for k in k_values},
        'recall': {k: [] for k in k_values}
    }
    
    # Process each query
    for query_id, group in query_groups:
        if query_id not in ground_truth_groups.groups:
            logger.warning(f"Query {query_id} not found in ground truth")
            continue
        
        # Get ground truth for this query
        gt_group = ground_truth_groups.get_group(query_id)
        
        # Create a mapping from product_id to gain
        product_to_gain = {row['product_id']: row['gain'] for _, row in gt_group.iterrows()}
        
        # Get ranked products
        ranked_products = group['product_id'].tolist()
        
        # Create arrays for evaluation
        y_true = np.array([product_to_gain.get(pid, 0.0) for pid in ranked_products])
        y_score = np.array(range(len(ranked_products), 0, -1))  # Use rank as score
        
        # Compute metrics
        for k in k_values:
            if len(y_true) >= k:
                # NDCG@k
                ndcg = compute_ndcg(y_true, y_score, k=k)
                metrics['ndcg'][k].append(ndcg)
                
                # Recall@k
                recall = compute_recall_at_k(y_true, y_score, k=k)
                metrics['recall'][k].append(recall)
    
    # Average metrics across queries
    results = {}
    for metric_name, k_dict in metrics.items():
        for k, values in k_dict.items():
            if values:
                results[f'{metric_name}@{k}'] = np.mean(values)
    
    return results
