"""
Command-line script for training a two-tower model
"""

import os
import argparse
import torch
import random
import numpy as np
from transformers import AutoTokenizer
import logging

from esci_ranking.models.two_tower import create_model
from esci_ranking.data.dataset import load_data, create_data_loaders
from esci_ranking.train.trainer import TwoTowerTrainer
from esci_ranking.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description='Train a two-tower model for query-product matching')
    parser.add_argument('train_csv_path', type=str, help='Path to the training CSV file')
    parser.add_argument('product_catalog_csv_path', type=str, help='Path to the product catalog CSV file')
    parser.add_argument('model_save_path', type=str, help='Directory to save the model')
    parser.add_argument('--locale', type=str, default='us', choices=['us', 'es', 'jp'], help='Locale of the queries')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Base embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=256, help='Projection dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased', help='Pre-trained model name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    args = parser.parse_args()
    
    # Set up logging
    log_dir = os.path.join(args.model_save_path, args.log_dir)
    logger = setup_logging(log_dir=log_dir)
    
    # Log arguments
    logger.info("Training with the following parameters:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load and prepare data
    train_df, val_df, product_catalog_df, _ = load_data(
        args.train_csv_path, 
        args.product_catalog_csv_path,
        args.locale
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create data loaders
    train_dataloader, val_dataloader, _ = create_data_loaders(
        train_df,
        val_df,
        product_catalog_df,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = create_model(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # Create trainer
    trainer = TwoTowerTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        model_save_path=args.model_save_path
    )
    
    # Train model
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    model, train_losses, val_metrics = trainer.train(num_epochs=args.num_epochs)
    
    # Save tokenizer
    tokenizer.save_pretrained(args.model_save_path)
    logger.info(f"Saved tokenizer to {args.model_save_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
