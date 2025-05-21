"""
Example script for running inference with a trained two-tower model
"""

import os
import torch
from transformers import AutoTokenizer
import logging

from esci_ranking.models.two_tower import load_inference_encoders
from esci_ranking.inference.predictor import TwoTowerPredictor
from esci_ranking.utils.logging_utils import setup_logging
from esci_ranking.utils.data_utils import save_rankings

# Set up logging
logger = setup_logging()

# Define paths
data_dir = "/Users/mgartner/Documents/personal_projects/mle_competitions/esci-data/data"
test_csv_path = os.path.join(data_dir, "processed", "public", "task_1_query-product_ranking", "test_public-v0.3.csv")
product_catalog_csv_path = os.path.join(data_dir, "processed", "public", "task_1_query-product_ranking", "product_catalogue-v0.3.csv")
model_path = os.path.join(data_dir, "models", "two_tower_model")
output_path = os.path.join(data_dir, "predictions", "two_tower_predictions.csv")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load data
import pandas as pd
logger.info(f"Loading test data from {test_csv_path}...")
test_df = pd.read_csv(test_csv_path)

logger.info(f"Loading product catalog from {product_catalog_csv_path}...")
product_catalog_df = pd.read_csv(product_catalog_csv_path)

# Filter by locale
locale = 'us'
test_df = test_df[test_df['query_locale'] == locale]
product_catalog_df = product_catalog_df[product_catalog_df['product_locale'] == locale]

logger.info(f"Test data: {len(test_df)} rows, {test_df['query_id'].nunique()} unique queries")
logger.info(f"Product catalog: {len(product_catalog_df)} products")

# Load tokenizer and model components
logger.info("Loading model components...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create embeddings directory
embeddings_dir = os.path.join(model_path, "embeddings")
os.makedirs(embeddings_dir, exist_ok=True)

# Load encoders
query_encoder, product_encoder = load_inference_encoders(
    model_path,
    embedding_dim=768,
    projection_dim=256
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
    batch_size=32,
    max_length=256,
    embeddings_dir=embeddings_dir,
    locale=locale
)

# Generate rankings
logger.info("Generating rankings...")
rankings_df = predictor.predict(
    test_df=test_df,
    product_catalog_df=product_catalog_df,
    use_cached_embeddings=False  # Set to True to use cached embeddings if available
)

# Save rankings
logger.info(f"Saving rankings to {output_path}")

# Keep only required columns for submission
if 'score' in rankings_df.columns:
    output_df = rankings_df[['query_id', 'product_id']]
else:
    output_df = rankings_df

# Save to CSV
save_rankings(output_df, output_path)

logger.info("Inference completed successfully!")
