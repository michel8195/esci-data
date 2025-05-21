"""
Command-line script for running inference with a trained two-tower model
"""

import os
import argparse
import torch
from transformers import AutoTokenizer
import logging

from esci_ranking.models.two_tower import load_inference_encoders
from esci_ranking.inference.predictor import TwoTowerPredictor
from esci_ranking.utils.logging_utils import setup_logging
from esci_ranking.utils.data_utils import save_rankings


def main():
    parser = argparse.ArgumentParser(description='Generate rankings using a trained two-tower model')
    parser.add_argument('test_csv_path', type=str, help='Path to the test CSV file')
    parser.add_argument('product_catalog_csv_path', type=str, help='Path to the product catalog CSV file')
    parser.add_argument('model_path', type=str, help='Path to the trained model directory')
    parser.add_argument('output_path', type=str, help='Path to save the output rankings')
    parser.add_argument('--locale', type=str, default=None, choices=['us', 'es', 'jp'], help='Locale of the queries')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Base embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=256, help='Projection dimension')
    parser.add_argument('--use_cached_embeddings', action='store_true', help='Use cached embeddings if available')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    args = parser.parse_args()
    
    # Set up logging
    log_dir = os.path.join(args.model_path, args.log_dir)
    logger = setup_logging(log_dir=log_dir)
    
    # Log arguments
    logger.info("Running inference with the following parameters:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    import pandas as pd
    logger.info(f"Loading test data from {args.test_csv_path}...")
    test_df = pd.read_csv(args.test_csv_path)
    
    logger.info(f"Loading product catalog from {args.product_catalog_csv_path}...")
    product_catalog_df = pd.read_csv(args.product_catalog_csv_path)
    
    # Filter by locale if specified
    if args.locale:
        test_df = test_df[test_df['query_locale'] == args.locale]
        product_catalog_df = product_catalog_df[product_catalog_df['product_locale'] == args.locale]
    
    logger.info(f"Test data: {len(test_df)} rows, {test_df['query_id'].nunique()} unique queries")
    logger.info(f"Product catalog: {len(product_catalog_df)} products")
    
    # Load tokenizer and model components
    logger.info("Loading model components...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Create embeddings directory
    embeddings_dir = os.path.join(args.model_path, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Load encoders
    query_encoder, product_encoder = load_inference_encoders(
        args.model_path,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim
    )
    
    # Move encoders to device
    query_encoder = query_encoder.to(device)
    product_encoder = product_encoder.to(device)
    
    # Create predictor
    predictor = TwoTowerPredictor(
        query_encoder=query_encoder,
        product_encoder=product_encoder,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        embeddings_dir=embeddings_dir,
        locale=args.locale
    )
    
    # Generate rankings
    logger.info("Generating rankings...")
    rankings_df = predictor.predict(
        test_df=test_df,
        product_catalog_df=product_catalog_df,
        use_cached_embeddings=args.use_cached_embeddings
    )
    
    # Save rankings
    logger.info(f"Saving rankings to {args.output_path}")
    
    # Keep only required columns for submission
    if 'score' in rankings_df.columns:
        output_df = rankings_df[['query_id', 'product_id']]
    else:
        output_df = rankings_df
    
    # Save to CSV
    save_rankings(output_df, args.output_path)
    
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
