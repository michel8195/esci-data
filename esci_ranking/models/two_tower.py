"""
Two-Tower model architecture for query-product matching
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)


class TwoTowerModel(nn.Module):
    """
    Two-Tower model for query-product matching
    
    Args:
        query_encoder: Transformer model for encoding queries
        product_encoder: Transformer model for encoding products
        embedding_dim: Dimension of the base encoder embeddings
        projection_dim: Dimension of the projection space
        dropout_rate: Dropout rate for regularization
    """
    def __init__(
        self, 
        query_encoder, 
        product_encoder, 
        embedding_dim=768, 
        projection_dim=256,
        dropout_rate=0.1
    ):
        super(TwoTowerModel, self).__init__()
        
        # Base encoders
        self.query_encoder = query_encoder
        self.product_encoder = product_encoder
        
        # Projection layers for query tower
        self.query_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Projection layers for product tower
        self.product_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def encode_query(self, input_ids, attention_mask):
        """
        Encode query text into embeddings
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Normalized query embeddings
        """
        outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0]  # Use CLS token embedding
        query_embeddings = self.query_projection(embeddings)
        return F.normalize(query_embeddings, p=2, dim=1)
    
    def encode_product(self, input_ids, attention_mask):
        """
        Encode product text into embeddings
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Normalized product embeddings
        """
        outputs = self.product_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0]  # Use CLS token embedding
        product_embeddings = self.product_projection(embeddings)
        return F.normalize(product_embeddings, p=2, dim=1)
    
    def forward(self, query_input_ids, query_attention_mask, product_input_ids, product_attention_mask):
        """
        Forward pass to compute similarity scores
        
        Args:
            query_input_ids: Query token IDs
            query_attention_mask: Query attention mask
            product_input_ids: Product token IDs
            product_attention_mask: Product attention mask
            
        Returns:
            Similarity matrix between queries and products
        """
        query_embeddings = self.encode_query(query_input_ids, query_attention_mask)
        product_embeddings = self.encode_product(product_input_ids, product_attention_mask)
        similarity = torch.mm(query_embeddings, product_embeddings.transpose(0, 1))
        return similarity


class QueryEncoder(nn.Module):
    """
    Query encoder with projection layers for inference
    
    Args:
        encoder_model: Base transformer model
        projection_path: Path to saved projection weights
        embedding_dim: Dimension of the base encoder embeddings
        projection_dim: Dimension of the projection space
    """
    def __init__(self, encoder_model, projection_path, embedding_dim=768, projection_dim=256):
        super(QueryEncoder, self).__init__()
        self.encoder = encoder_model
        
        # Load projection layers
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Load saved weights
        self.projection.load_state_dict(torch.load(projection_path))
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass to compute query embeddings
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Normalized query embeddings
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0]  # Use CLS token embedding
        query_embeddings = self.projection(embeddings)
        return F.normalize(query_embeddings, p=2, dim=1)


class ProductEncoder(nn.Module):
    """
    Product encoder with projection layers for inference
    
    Args:
        encoder_model: Base transformer model
        projection_path: Path to saved projection weights
        embedding_dim: Dimension of the base encoder embeddings
        projection_dim: Dimension of the projection space
    """
    def __init__(self, encoder_model, projection_path, embedding_dim=768, projection_dim=256):
        super(ProductEncoder, self).__init__()
        self.encoder = encoder_model
        
        # Load projection layers
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Load saved weights
        self.projection.load_state_dict(torch.load(projection_path))
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass to compute product embeddings
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Normalized product embeddings
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0]  # Use CLS token embedding
        product_embeddings = self.projection(embeddings)
        return F.normalize(product_embeddings, p=2, dim=1)


def create_model(model_name='bert-base-multilingual-cased', embedding_dim=768, projection_dim=256, dropout_rate=0.1):
    """
    Create a new two-tower model
    
    Args:
        model_name: Name of the pre-trained model to use
        embedding_dim: Dimension of the base encoder embeddings
        projection_dim: Dimension of the projection space
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Initialized TwoTowerModel
    """
    # Initialize encoders
    query_encoder = AutoModel.from_pretrained(model_name)
    product_encoder = AutoModel.from_pretrained(model_name)
    
    # Create model
    model = TwoTowerModel(
        query_encoder=query_encoder,
        product_encoder=product_encoder,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        dropout_rate=dropout_rate
    )
    
    return model


def load_inference_encoders(model_path, embedding_dim=768, projection_dim=256):
    """
    Load query and product encoders for inference
    
    Args:
        model_path: Path to the saved model
        embedding_dim: Dimension of the base encoder embeddings
        projection_dim: Dimension of the projection space
        
    Returns:
        Tuple of (query_encoder, product_encoder)
    """
    # Load query encoder
    query_encoder_path = os.path.join(model_path, "query_encoder")
    query_base_encoder = AutoModel.from_pretrained(query_encoder_path)
    query_projection_path = os.path.join(query_encoder_path, "projection.pt")
    
    query_encoder = QueryEncoder(
        encoder_model=query_base_encoder,
        projection_path=query_projection_path,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim
    )
    
    # Load product encoder
    product_encoder_path = os.path.join(model_path, "product_encoder")
    product_base_encoder = AutoModel.from_pretrained(product_encoder_path)
    product_projection_path = os.path.join(product_encoder_path, "projection.pt")
    
    product_encoder = ProductEncoder(
        encoder_model=product_base_encoder,
        projection_path=product_projection_path,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim
    )
    
    return query_encoder, product_encoder
