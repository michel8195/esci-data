"""
Training utilities for the two-tower model
"""

import os
import time
import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt

from esci_ranking.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class TwoTowerTrainer:
    """
    Trainer for the two-tower model
    
    Args:
        model: The two-tower model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        device: Device to train on
        learning_rate: Learning rate for optimization
        warmup_ratio: Ratio of warmup steps
        model_save_path: Path to save the model
    """
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        device,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        model_save_path=None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.model_save_path = model_save_path
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Calculate total steps
        self.total_steps = len(self.train_dataloader) * 3  # Default to 3 epochs
        self.warmup_steps = int(self.total_steps * self.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=self.warmup_steps, 
            num_training_steps=self.total_steps
        )
        
        # Training metrics
        self.train_losses = []
        self.val_metrics_history = []
        self.best_val_ndcg = 0.0
    
    def train(self, num_epochs=3, fp16=False, gradient_accumulation_steps=1):
        """
        Train the model for the specified number of epochs
        
        Args:
            num_epochs: Number of training epochs
            fp16: Whether to use mixed precision training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Tuple of (trained model, training losses, validation metrics)
        """
        logger.info("Starting training...")
        
        # Update total steps and scheduler if num_epochs changed
        if num_epochs * len(self.train_dataloader) != self.total_steps:
            self.total_steps = len(self.train_dataloader) * num_epochs // gradient_accumulation_steps
            self.warmup_steps = int(self.total_steps * self.warmup_ratio)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.warmup_steps, 
                num_training_steps=self.total_steps
            )
        
        # Set up mixed precision training if requested
        scaler = None
        if fp16 and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            logger.info("Using mixed precision training (FP16)")
        
        # Calculate total examples for reporting
        total_examples = sum(len(batch['query_ids']) for batch in self.train_dataloader)
        logger.info(f"Training on {total_examples} examples with {len(self.train_dataloader)} batches")
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            epoch_loss = 0
            epoch_start_time = time.time()
            examples_processed = 0
            running_loss = 0
            last_log_time = time.time()
            
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f"Training epoch {epoch+1}")):
                # Move batch to device
                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device)
                product_input_ids = batch['product_input_ids'].to(self.device)
                product_attention_mask = batch['product_attention_mask'].to(self.device)
                gains = batch['gains'].to(self.device)
                num_products = batch['num_products']
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Process each query with its products
                start_idx = 0
                batch_loss = 0
                
                for i, num_prods in enumerate(num_products):
                    # Get query and its products
                    query_input = query_input_ids[i].unsqueeze(0)
                    query_mask = query_attention_mask[i].unsqueeze(0)
                    
                    product_input = product_input_ids[start_idx:start_idx + num_prods]
                    product_mask = product_attention_mask[start_idx:start_idx + num_prods]
                    product_gains = gains[start_idx:start_idx + num_prods]
                    
                    # Encode query and products
                    query_emb = self.model.encode_query(query_input, query_mask)
                    product_emb = self.model.encode_product(product_input, product_mask)
                    
                    # Compute similarity scores
                    similarity = torch.mm(query_emb, product_emb.transpose(0, 1)).squeeze()
                    
                    # Compute loss (MSE between similarity scores and gain values)
                    loss = F.mse_loss(similarity, product_gains)
                    batch_loss += loss
                    
                    start_idx += num_prods
                
                # Average loss across queries in the batch
                batch_loss = batch_loss / len(num_products)
                
                # Scale loss by gradient accumulation steps
                batch_loss = batch_loss / gradient_accumulation_steps
                
                # Backward pass with mixed precision if enabled
                if scaler is not None:
                    scaler.scale(batch_loss).backward()
                else:
                    batch_loss.backward()
                
                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Track loss and examples processed
                examples_processed += len(num_products)
                running_loss += batch_loss.item() * gradient_accumulation_steps
                epoch_loss += batch_loss.item() * gradient_accumulation_steps
                
                # Log training speed every 10 seconds
                current_time = time.time()
                if current_time - last_log_time > 10:
                    elapsed = current_time - epoch_start_time
                    examples_per_second = examples_processed / elapsed
                    avg_loss = running_loss / (batch_idx + 1)
                    logger.info(f"Processed {examples_processed}/{total_examples} examples "
                                f"({examples_per_second:.2f} examples/sec) - "
                                f"Loss: {avg_loss:.4f}")
                    last_log_time = current_time
            
            # Calculate and log epoch statistics
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            examples_per_second = examples_processed / epoch_time
            self.train_losses.append(avg_epoch_loss)
            
            logger.info(f"Epoch {epoch+1} - "
                        f"Average training loss: {avg_epoch_loss:.4f} - "
                        f"Training speed: {examples_per_second:.2f} examples/sec - "
                        f"Time: {epoch_time:.2f}s")
            
            # Validation phase
            val_metrics = self.evaluate()
            self.val_metrics_history.append(val_metrics)
            
            # Log validation metrics
            logger.info(f"Validation metrics:")
            for metric, value in val_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Save best model
            if val_metrics.get('ndcg@10', 0) > self.best_val_ndcg and self.model_save_path:
                self.best_val_ndcg = val_metrics.get('ndcg@10', 0)
                logger.info(f"New best NDCG@10: {self.best_val_ndcg:.4f} - Saving model")
                
                # Create directory if it doesn't exist
                os.makedirs(self.model_save_path, exist_ok=True)
                
                # Save model state
                torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "best_model.pt"))
        
        # Save final model
        if self.model_save_path:
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "final_model.pt"))
            
            # Save query and product encoders separately for inference
            self._save_encoders()
            
            # Plot training curves
            self._plot_training_curves()
        
        return self.model, self.train_losses, self.val_metrics_history
    
    def evaluate(self):
        """
        Evaluate the model on the validation set
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        all_scores = []
        all_gains = []
        all_num_products = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # Move batch to device
                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device)
                product_input_ids = batch['product_input_ids'].to(self.device)
                product_attention_mask = batch['product_attention_mask'].to(self.device)
                gains = batch['gains'].to(self.device)
                num_products = batch['num_products']
                
                # Process each query with its products
                start_idx = 0
                
                for i, num_prods in enumerate(num_products):
                    # Get query and its products
                    query_input = query_input_ids[i].unsqueeze(0)
                    query_mask = query_attention_mask[i].unsqueeze(0)
                    
                    product_input = product_input_ids[start_idx:start_idx + num_prods]
                    product_mask = product_attention_mask[start_idx:start_idx + num_prods]
                    product_gains = gains[start_idx:start_idx + num_prods]
                    
                    # Encode query and products
                    query_emb = self.model.encode_query(query_input, query_mask)
                    product_emb = self.model.encode_product(product_input, product_mask)
                    
                    # Compute similarity scores
                    similarity = torch.mm(query_emb, product_emb.transpose(0, 1)).squeeze()
                    
                    # Store scores and gains
                    all_scores.append(similarity)
                    all_gains.append(product_gains)
                    all_num_products.append(num_prods)
                    
                    start_idx += num_prods
        
        # Concatenate results
        all_scores = torch.cat(all_scores)
        all_gains = torch.cat(all_gains)
        
        # Compute metrics
        metrics = compute_metrics(all_scores, all_gains, all_num_products)
        
        return metrics
    
    def _save_encoders(self):
        """Save query and product encoders separately for inference"""
        if not self.model_save_path:
            return
        
        # Create directories
        query_encoder_path = os.path.join(self.model_save_path, "query_encoder")
        product_encoder_path = os.path.join(self.model_save_path, "product_encoder")
        
        os.makedirs(query_encoder_path, exist_ok=True)
        os.makedirs(product_encoder_path, exist_ok=True)
        
        # Save the query encoder
        self.model.query_encoder.save_pretrained(query_encoder_path)
        torch.save(self.model.query_projection.state_dict(), os.path.join(query_encoder_path, "projection.pt"))
        
        # Save the product encoder
        self.model.product_encoder.save_pretrained(product_encoder_path)
        torch.save(self.model.product_projection.state_dict(), os.path.join(product_encoder_path, "projection.pt"))
        
        logger.info(f"Saved encoders to {self.model_save_path}")
    
    def _plot_training_curves(self):
        """Plot training and validation curves"""
        if not self.model_save_path:
            return
        
        # Create figure directory
        fig_dir = os.path.join(self.model_save_path, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, 'training_loss.png'))
        plt.close()
        
        # Plot validation metrics
        for metric in self.val_metrics_history[0].keys():
            values = [metrics.get(metric, 0) for metrics in self.val_metrics_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(values, label=f'Validation {metric}')
            plt.title(f'Validation {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(fig_dir, f'validation_{metric}.png'))
            plt.close()
        
        logger.info(f"Saved training curves to {fig_dir}")
