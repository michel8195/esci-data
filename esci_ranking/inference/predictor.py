"""
Inference utilities for the two-tower model
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def encode_batch(texts, tokenizer, encoder, device, batch_size=32, max_length=256):
    """
    Encode a batch of texts and return embeddings
    
    Args:
        texts: List of texts to encode
        tokenizer: Tokenizer for encoding text
        encoder: Encoder model
        device: Device to run encoding on
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        
    Returns:
        Tensor of embeddings
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        # Prepare batch
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        
        # Compute embeddings
        with torch.no_grad():
            embeddings = encoder(
                input_ids=encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask']
            )
        
        all_embeddings.append(embeddings.cpu())
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    return all_embeddings


def create_product_texts(product_catalog_df):
    """
    Create rich text representations of products by combining fields
    
    Args:
        product_catalog_df: DataFrame containing product catalog information
        
    Returns:
        Dictionary mapping product IDs to text representations
    """
    product_texts = {}
    
    for _, row in tqdm(product_catalog_df.iterrows(), total=len(product_catalog_df), desc="Processing products"):
        product_id = row['product_id']
        
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
        
        product_texts[product_id] = product_text
    
    return product_texts


class TwoTowerPredictor:
    """
    Predictor for the two-tower model
    
    Args:
        query_encoder: Query encoder model
        product_encoder: Product encoder model
        tokenizer: Tokenizer for encoding text
        device: Device to run inference on
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        embeddings_dir: Directory to cache embeddings
        locale: Locale for the data
    """
    def __init__(
        self,
        query_encoder,
        product_encoder,
        tokenizer,
        device,
        batch_size=32,
        max_length=256,
        embeddings_dir=None,
        locale=None
    ):
        self.query_encoder = query_encoder
        self.product_encoder = product_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.embeddings_dir = embeddings_dir
        self.locale = locale
        
        # Set models to evaluation mode
        self.query_encoder.eval()
        self.product_encoder.eval()
        
        # Create embeddings directory if specified
        if self.embeddings_dir:
            os.makedirs(self.embeddings_dir, exist_ok=True)
    
    def encode_queries(self, queries, use_cached=False):
        """
        Encode queries
        
        Args:
            queries: List of query texts
            use_cached: Whether to use cached embeddings if available
            
        Returns:
            Tensor of query embeddings
        """
        # Define path for cached embeddings
        cache_path = None
        if self.embeddings_dir:
            cache_path = os.path.join(
                self.embeddings_dir, 
                f"query_embeddings{'_' + self.locale if self.locale else ''}.pt"
            )
        
        # Load cached embeddings if available and requested
        if use_cached and cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached query embeddings from {cache_path}")
            return torch.load(cache_path)
        
        # Encode queries
        logger.info("Encoding queries...")
        query_embeddings = encode_batch(
            texts=queries,
            tokenizer=self.tokenizer,
            encoder=self.query_encoder,
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length
        )
        
        # Cache embeddings if requested
        if cache_path:
            logger.info(f"Caching query embeddings to {cache_path}")
            torch.save(query_embeddings, cache_path)
        
        return query_embeddings
    
    def encode_products(self, product_texts, use_cached=False):
        """
        Encode products
        
        Args:
            product_texts: List of product text representations
            use_cached: Whether to use cached embeddings if available
            
        Returns:
            Tensor of product embeddings
        """
        # Define path for cached embeddings
        cache_path = None
        if self.embeddings_dir:
            cache_path = os.path.join(
                self.embeddings_dir, 
                f"product_embeddings{'_' + self.locale if self.locale else ''}.pt"
            )
        
        # Load cached embeddings if available and requested
        if use_cached and cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached product embeddings from {cache_path}")
            return torch.load(cache_path)
        
        # Encode products
        logger.info("Encoding products...")
        product_embeddings = encode_batch(
            texts=product_texts,
            tokenizer=self.tokenizer,
            encoder=self.product_encoder,
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length
        )
        
        # Cache embeddings if requested
        if cache_path:
            logger.info(f"Caching product embeddings to {cache_path}")
            torch.save(product_embeddings, cache_path)
        
        return product_embeddings
    
    def predict(self, test_df, product_catalog_df, use_cached_embeddings=False):
        """
        Generate rankings for test queries
        
        Args:
            test_df: DataFrame containing test queries
            product_catalog_df: DataFrame containing product catalog
            use_cached_embeddings: Whether to use cached embeddings if available
            
        Returns:
            DataFrame with rankings (query_id, product_id)
        """
        # Get unique queries
        unique_queries = test_df[['query_id', 'query']].drop_duplicates()
        
        # Get unique products from test set
        test_product_ids = test_df['product_id'].unique()
        
        # Filter product catalog to only include products in the test set
        filtered_product_catalog = product_catalog_df[product_catalog_df['product_id'].isin(test_product_ids)]
        logger.info(f"Filtered product catalog to {len(filtered_product_catalog)} products found in test set")
        
        # Create product text representations
        product_texts = create_product_texts(filtered_product_catalog)
        
        # Create mapping from product ID to index
        product_id_to_idx = {pid: i for i, pid in enumerate(filtered_product_catalog['product_id'])}
        
        # Encode queries
        query_texts = unique_queries['query'].tolist()
        query_embeddings = self.encode_queries(query_texts, use_cached=use_cached_embeddings)
        
        # Get text representations for products in the test set
        product_text_list = []
        product_ids = []
        
        for pid in test_product_ids:
            if pid in product_texts:
                product_text_list.append(product_texts[pid])
                product_ids.append(pid)
            else:
                # If product not found in catalog, use product ID as fallback
                logger.warning(f"Product {pid} not found in catalog, using ID as text")
                product_text_list.append(f"Product ID: {pid}")
                product_ids.append(pid)
        
        # Encode products
        product_embeddings = self.encode_products(product_text_list, use_cached=use_cached_embeddings)
        
        logger.info("Computing similarity scores and generating rankings...")
        
        # Group test data by query
        query_groups = test_df.groupby('query_id')
        
        # Prepare output data
        output_rows = []
        
        # Process each query
        for i, (_, query_row) in enumerate(tqdm(unique_queries.iterrows(), total=len(unique_queries), desc="Ranking")):
            query_id = query_row['query_id']
            query_embedding = query_embeddings[i].unsqueeze(0)  # Shape: [1, embedding_dim]
            
            # Get products for this query
            query_products = query_groups.get_group(query_id)['product_id'].tolist()
            
            # Get embeddings for these products
            product_indices = [product_id_to_idx.get(pid, -1) for pid in query_products]
            valid_indices = [i for i in product_indices if i != -1]
            
            if not valid_indices:
                logger.warning(f"No valid products found for query {query_id}")
                continue
            
            query_product_embeddings = product_embeddings[valid_indices]
            
            # Compute similarity scores
            similarity_scores = torch.mm(query_embedding, query_product_embeddings.t()).squeeze().tolist()
            
            # Create ranking
            valid_products = [query_products[i] for i, idx in enumerate(product_indices) if idx != -1]
            ranking = list(zip(valid_products, similarity_scores))
            ranking.sort(key=lambda x: x[1], reverse=True)  # Sort by score in descending order
            
            # Add to output
            for product_id, score in ranking:
                output_rows.append({
                    'query_id': query_id, 
                    'product_id': product_id, 
                    'score': score
                })
        
        # Create output dataframe
        output_df = pd.DataFrame(output_rows)
        
        return output_df
