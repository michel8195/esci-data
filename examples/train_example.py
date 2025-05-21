"""
Example script for training a two-tower model using the ESCI Ranking package
"""

import os
import torch
from transformers import AutoTokenizer
import logging

from esci_ranking.models.two_tower import create_model
from esci_ranking.data.dataset import load_data, create_data_loaders
from esci_ranking.train.trainer import TwoTowerTrainer
from esci_ranking.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging()

# Define paths
data_dir = "/Users/mgartner/Documents/personal_projects/mle_competitions/esci-data/data"
train_csv_path = os.path.join(data_dir, "processed", "public", "task_1_query-product_ranking", "train-v0.3.csv")
product_catalog_csv_path = os.path.join(data_dir, "processed", "public", "task_1_query-product_ranking", "product_catalogue-v0.3.csv")
model_save_path = os.path.join(data_dir, "models", "two_tower_model")

# Create model directory if it doesn't exist
os.makedirs(model_save_path, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load and prepare data
train_df, val_df, product_catalog_df, _ = load_data(
    train_csv_path, 
    product_catalog_csv_path,
    locale='us'
)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# Create data loaders
train_dataloader, val_dataloader, _ = create_data_loaders(
    train_df,
    val_df,
    product_catalog_df,
    tokenizer,
    batch_size=4,
    max_length=256
)

# Initialize model
logger.info("Initializing model...")
model = create_model(
    model_name='bert-base-multilingual-cased',
    embedding_dim=768,
    projection_dim=256,
    dropout_rate=0.1
).to(device)

# Create trainer
trainer = TwoTowerTrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device=device,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    model_save_path=model_save_path
)

# Train model with acceleration techniques
logger.info("Starting training with acceleration techniques...")

# Set training parameters
num_epochs = 3
fp16 = torch.cuda.is_available()  # Use mixed precision if GPU is available
gradient_accumulation_steps = 4  # Accumulate gradients to simulate larger batch size

model, train_losses, val_metrics = trainer.train(
    num_epochs=num_epochs,
    fp16=fp16,
    gradient_accumulation_steps=gradient_accumulation_steps
)

# Save tokenizer
tokenizer.save_pretrained(model_save_path)
logger.info(f"Saved tokenizer to {model_save_path}")

logger.info("Training completed successfully!")
