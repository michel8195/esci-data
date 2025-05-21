"""
Data processing utilities for the ESCI ranking package
"""

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


def filter_by_locale(df, locale, locale_column='product_locale'):
    """
    Filter DataFrame by locale
    
    Args:
        df: DataFrame to filter
        locale: Locale to filter by
        locale_column: Column containing locale information
        
    Returns:
        Filtered DataFrame
    """
    if locale and locale_column in df.columns:
        filtered_df = df[df[locale_column] == locale]
        logger.info(f"Filtered {len(df)} rows to {len(filtered_df)} rows with locale {locale}")
        return filtered_df
    return df


def merge_query_product_data(query_df, product_catalog_df):
    """
    Merge query data with product catalog
    
    Args:
        query_df: DataFrame containing query information
        product_catalog_df: DataFrame containing product catalog information
        
    Returns:
        Merged DataFrame
    """
    merged_df = pd.merge(
        query_df,
        product_catalog_df,
        on='product_id',
        how='left'
    )
    
    # Check for missing products
    missing_products = merged_df[merged_df['product_title'].isna()]['product_id'].nunique()
    if missing_products > 0:
        logger.warning(f"Found {missing_products} products in query data not present in product catalog")
    
    return merged_df


def create_product_representation(row):
    """
    Create a rich text representation of a product by combining fields
    
    Args:
        row: DataFrame row containing product information
        
    Returns:
        Text representation of the product
    """
    # Start with the product title
    product_text = f"Title: {row['product_title']}"
    
    # Add brand if available
    if pd.notna(row.get('product_brand')):
        product_text += f" Brand: {row['product_brand']}"
    
    # Add color if available
    if pd.notna(row.get('product_color_name')):
        product_text += f" Color: {row['product_color_name']}"
    
    # Add bullet points if available
    if pd.notna(row.get('product_bullet_point')):
        bullet_points = str(row['product_bullet_point'])
        if len(bullet_points) > 0:
            # Truncate bullet points if too long
            if len(bullet_points) > 500:
                bullet_points = bullet_points[:500] + "..."
            product_text += f" Features: {bullet_points}"
    
    # Add description if available
    if pd.notna(row.get('product_description')):
        description = str(row['product_description'])
        if len(description) > 0:
            # Truncate description if too long
            if len(description) > 300:
                description = description[:300] + "..."
            product_text += f" Description: {description}"
    
    return product_text


def save_rankings(rankings_df, output_path):
    """
    Save rankings to CSV file
    
    Args:
        rankings_df: DataFrame containing rankings
        output_path: Path to save the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    rankings_df.to_csv(output_path, index=False)
    logger.info(f"Saved rankings to {output_path}")


def load_esci_data(data_dir, locale=None):
    """
    Load ESCI data from the standard directory structure
    
    Args:
        data_dir: Base directory containing the data
        locale: Optional locale to filter data
        
    Returns:
        Dictionary containing DataFrames for train, test, and product catalog
    """
    # Define paths
    train_path = os.path.join(data_dir, 'processed', 'public', 'task_1_query-product_ranking', 'train-v0.3.csv')
    test_path = os.path.join(data_dir, 'processed', 'public', 'task_1_query-product_ranking', 'test_public-v0.3.csv')
    product_catalog_path = os.path.join(data_dir, 'processed', 'public', 'task_1_query-product_ranking', 'product_catalogue-v0.3.csv')
    
    # Load data
    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loading product catalog from {product_catalog_path}")
    product_catalog_df = pd.read_csv(product_catalog_path)
    
    # Filter by locale if specified
    if locale:
        train_df = filter_by_locale(train_df, locale, 'query_locale')
        test_df = filter_by_locale(test_df, locale, 'query_locale')
        product_catalog_df = filter_by_locale(product_catalog_df, locale)
    
    return {
        'train': train_df,
        'test': test_df,
        'product_catalog': product_catalog_df
    }
