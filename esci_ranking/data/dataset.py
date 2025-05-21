"""
Dataset classes and data loading utilities for ESCI ranking
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class ESCIDataset(Dataset):
    """
    Dataset for ESCI query-product pairs with product catalog information

    Args:
        query_df: DataFrame containing query information
        product_catalog_df: DataFrame containing product catalog information
        tokenizer: Tokenizer for encoding text
        max_length: Maximum sequence length for tokenization
        is_training: Whether the dataset is used for training
    """
    def __init__(self, query_df, product_catalog_df, tokenizer, max_length=128, is_training=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training

        # Merge query data with product catalog
        self.df = pd.merge(
            query_df,
            product_catalog_df,
            on='product_id',
            how='left'
        )

        # Group data by query_id
        self.query_groups = {}
        for query_id, group in self.df.groupby('query_id'):
            self.query_groups[query_id] = group.reset_index(drop=True)

        self.query_ids = list(self.query_groups.keys())

        # ESCI label to gain mapping
        self.esci_label2gain = {
            'exact': 1.0,
            'substitute': 0.1,
            'complement': 0.01,
            'irrelevant': 0.0,
        }

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        group = self.query_groups[query_id]

        # Get query text
        query = group['query'].iloc[0]

        # Sample products for this query
        if self.is_training:
            # For training, sample a mix of relevant and irrelevant products
            if 'esci_label' in group.columns:
                exact_matches = group[group['esci_label'] == 'exact']
                substitutes = group[group['esci_label'] == 'substitute']
                complements = group[group['esci_label'] == 'complement']
                irrelevants = group[group['esci_label'] == 'irrelevant']

                # Sample products with different relevance levels
                sampled_products = []

                # Always include an exact match if available
                if len(exact_matches) > 0:
                    # Convert to dict to maintain consistency
                    sampled_products.extend(exact_matches.sample(min(1, len(exact_matches))).to_dict('records'))

                # Add some substitutes
                if len(substitutes) > 0:
                    sampled_products.extend(substitutes.sample(min(2, len(substitutes))).to_dict('records'))

                # Add a complement if available
                if len(complements) > 0:
                    # Convert to dict to maintain consistency
                    sampled_products.extend(complements.sample(min(1, len(complements))).to_dict('records'))

                # Add some irrelevant products
                if len(irrelevants) > 0:
                    sampled_products.extend(irrelevants.sample(min(2, len(irrelevants))).to_dict('records'))

                # If we don't have enough, sample randomly from the remaining products
                # Track indices of already sampled products
                sampled_indices = []
                for p in sampled_products:
                    if isinstance(p, dict):
                        # For products added as dictionaries (from to_dict('records'))
                        if 'index' in p:
                            sampled_indices.append(p['index'])
                    else:
                        # For products added as Series objects (from .iloc[0])
                        sampled_indices.append(p.name)

                remaining = group.loc[~group.index.isin(sampled_indices)]
                if len(sampled_products) < 6 and len(remaining) > 0:
                    sampled_products.extend(remaining.sample(min(6 - len(sampled_products), len(remaining))).to_dict('records'))

                # Convert list of dicts to dataframe for consistent processing
                if len(sampled_products) > 0:
                    products_df = pd.DataFrame(sampled_products)
                else:
                    products_df = group
            else:
                # For test data without labels, use all products
                products_df = group
        else:
            # For validation, use all products
            products_df = group

        # Create rich product representations by combining fields
        product_texts = []
        for _, row in products_df.iterrows():
            # Combine product fields into a single text representation
            product_text = f"Title: {row['product_title']}"

            if pd.notna(row.get('product_brand')):
                product_text += f" Brand: {row['product_brand']}"

            if pd.notna(row.get('product_color_name')):
                product_text += f" Color: {row['product_color_name']}"

            if pd.notna(row.get('product_bullet_point')):
                bullet_points = str(row['product_bullet_point'])
                if len(bullet_points) > 0:
                    # Truncate bullet points if too long
                    if len(bullet_points) > 500:
                        bullet_points = bullet_points[:500] + "..."
                    product_text += f" Features: {bullet_points}"

            product_texts.append(product_text)

        # Get labels if available
        if 'esci_label' in products_df.columns:
            esci_labels = products_df['esci_label'].tolist()
            gains = [self.esci_label2gain[label] for label in esci_labels]
        else:
            # For test data without labels, use dummy values
            gains = [0.0] * len(product_texts)

        # Tokenize query
        query_encoding = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize products
        product_encodings = [
            self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            for text in product_texts
        ]

        # Stack product encodings
        product_input_ids = torch.cat([enc['input_ids'] for enc in product_encodings])
        product_attention_mask = torch.cat([enc['attention_mask'] for enc in product_encodings])

        return {
            'query_id': query_id,
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'product_input_ids': product_input_ids.squeeze(),
            'product_attention_mask': product_attention_mask.squeeze(),
            'product_ids': products_df['product_id'].tolist(),
            'gains': torch.tensor(gains, dtype=torch.float),
            'num_products': len(product_texts)
        }


def collate_fn(batch):
    """
    Custom collate function for batching ESCI dataset items

    Args:
        batch: List of dataset items

    Returns:
        Batched data
    """
    query_ids = [item['query_id'] for item in batch]
    query_input_ids = torch.stack([item['query_input_ids'] for item in batch])
    query_attention_mask = torch.stack([item['query_attention_mask'] for item in batch])

    # Flatten product tensors
    all_product_input_ids = []
    all_product_attention_mask = []
    all_gains = []
    all_product_ids = []

    for item in batch:
        all_product_input_ids.append(item['product_input_ids'])
        all_product_attention_mask.append(item['product_attention_mask'])
        all_gains.append(item['gains'])
        all_product_ids.extend(item['product_ids'])

    # Stack or concatenate tensors
    if all(isinstance(x, torch.Tensor) and len(x.shape) == 2 for x in all_product_input_ids):
        all_product_input_ids = torch.cat(all_product_input_ids)
        all_product_attention_mask = torch.cat(all_product_attention_mask)
        all_gains = torch.cat(all_gains)
    else:
        # Handle case where some items might have only one product
        all_product_input_ids = torch.cat([x.unsqueeze(0) if len(x.shape) == 1 else x for x in all_product_input_ids])
        all_product_attention_mask = torch.cat([x.unsqueeze(0) if len(x.shape) == 1 else x for x in all_product_attention_mask])
        all_gains = torch.cat([x.unsqueeze(0) if len(x.shape) == 0 else x for x in all_gains])

    # Keep track of number of products per query for reshaping later
    num_products = [item['num_products'] for item in batch]

    return {
        'query_ids': query_ids,
        'query_input_ids': query_input_ids,
        'query_attention_mask': query_attention_mask,
        'product_input_ids': all_product_input_ids,
        'product_attention_mask': all_product_attention_mask,
        'product_ids': all_product_ids,
        'gains': all_gains,
        'num_products': num_products
    }


def load_data(train_csv_path, product_catalog_csv_path, locale=None, test_csv_path=None):
    """
    Load and prepare data from CSV files

    Args:
        train_csv_path: Path to the training CSV file
        product_catalog_csv_path: Path to the product catalog CSV file
        locale: Optional locale to filter data
        test_csv_path: Optional path to test CSV file

    Returns:
        Tuple of DataFrames (train_df, val_df, product_catalog_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    logger.info(f"Loading training data from {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path)

    logger.info(f"Loading product catalog from {product_catalog_csv_path}...")
    product_catalog_df = pd.read_csv(product_catalog_csv_path)

    # Load test data if provided
    test_df = None
    if test_csv_path:
        logger.info(f"Loading test data from {test_csv_path}...")
        test_df = pd.read_csv(test_csv_path)

    # Filter by locale if specified
    if locale:
        train_df = train_df[train_df['query_locale'] == locale]
        product_catalog_df = product_catalog_df[product_catalog_df['product_locale'] == locale]
        if test_df is not None:
            test_df = test_df[test_df['query_locale'] == locale]

    logger.info(f"Training data: {len(train_df)} rows, {train_df['query_id'].nunique()} unique queries")
    logger.info(f"Product catalog: {len(product_catalog_df)} products")

    # Split data by query_id for training
    query_ids = train_df['query_id'].unique()
    train_query_ids, val_query_ids = train_test_split(
        query_ids, test_size=0.1, random_state=42
    )

    # Create train and validation dataframes
    train_df_split = train_df[train_df['query_id'].isin(train_query_ids)]
    val_df_split = train_df[train_df['query_id'].isin(val_query_ids)]

    logger.info(f"Train set: {len(train_df_split)} rows, {train_df_split['query_id'].nunique()} unique queries")
    logger.info(f"Validation set: {len(val_df_split)} rows, {val_df_split['query_id'].nunique()} unique queries")
    if test_df is not None:
        logger.info(f"Test set: {len(test_df)} rows, {test_df['query_id'].nunique()} unique queries")

    return train_df_split, val_df_split, product_catalog_df, test_df


def create_data_loaders(train_df, val_df, product_catalog_df, tokenizer,
                        batch_size=4, max_length=256, test_df=None):
    """
    Create data loaders for training, validation, and testing

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        product_catalog_df: Product catalog DataFrame
        tokenizer: Tokenizer for encoding text
        batch_size: Batch size
        max_length: Maximum sequence length
        test_df: Optional test DataFrame

    Returns:
        Tuple of DataLoaders (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ESCIDataset(
        train_df,
        product_catalog_df,
        tokenizer,
        max_length=max_length,
        is_training=True
    )

    val_dataset = ESCIDataset(
        val_df,
        product_catalog_df,
        tokenizer,
        max_length=max_length,
        is_training=False
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create test dataloader if test data is provided
    test_dataloader = None
    if test_df is not None:
        test_dataset = ESCIDataset(
            test_df,
            product_catalog_df,
            tokenizer,
            max_length=max_length,
            is_training=False
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    return train_dataloader, val_dataloader, test_dataloader
