import os
import csv
import argparse
from collections import Counter, defaultdict
import numpy as np

def load_csv_data(file_path, max_rows=None):
    """Load data from CSV file"""
    print(f"Loading data from {file_path}...")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            data.append(row)
            if max_rows and i >= max_rows - 1:
                break
    
    print(f"Loaded {len(data)} rows")
    return data

def analyze_queries(data):
    """Analyze query information"""
    print("\n=== Query Analysis ===")
    
    # Count unique queries
    unique_queries = set()
    queries_by_locale = defaultdict(set)
    
    for row in data:
        query = row['query']
        query_id = row['query_id']
        locale = row['query_locale']
        
        unique_queries.add((query_id, query))
        queries_by_locale[locale].add((query_id, query))
    
    print(f"Total unique queries: {len(unique_queries)}")
    
    # Count queries by locale
    print("\nQueries by locale:")
    for locale, queries in queries_by_locale.items():
        print(f"{locale}: {len(queries)}")
    
    # Analyze query length
    query_lengths = [len(q[1]) for q in unique_queries]
    avg_length = sum(query_lengths) / len(query_lengths)
    min_length = min(query_lengths)
    max_length = max(query_lengths)
    
    print(f"\nQuery length statistics:")
    print(f"Average length: {avg_length:.2f} characters")
    print(f"Minimum length: {min_length} characters")
    print(f"Maximum length: {max_length} characters")
    
    # Sample queries
    print("\nSample queries (5 random):")
    import random
    sample_queries = random.sample(list(unique_queries), min(5, len(unique_queries)))
    for query_id, query in sample_queries:
        print(f"Query ID: {query_id}, Query: '{query}'")

def analyze_products(data):
    """Analyze product information"""
    print("\n=== Product Analysis ===")
    
    # Count unique products
    unique_products = set()
    products_by_locale = defaultdict(set)
    
    for row in data:
        product_id = row['product_id']
        locale = row['query_locale']  # Using query_locale as product_locale
        
        unique_products.add(product_id)
        products_by_locale[locale].add(product_id)
    
    print(f"Total unique products: {len(unique_products)}")
    
    # Count products by locale
    print("\nProducts by locale:")
    for locale, products in products_by_locale.items():
        print(f"{locale}: {len(products)}")

def analyze_query_product_relationships(data):
    """Analyze query-product relationships"""
    print("\n=== Query-Product Relationship Analysis ===")
    
    # Count products per query
    products_per_query = defaultdict(set)
    
    for row in data:
        query_id = row['query_id']
        product_id = row['product_id']
        
        products_per_query[query_id].add(product_id)
    
    # Calculate statistics
    products_counts = [len(products) for products in products_per_query.values()]
    avg_products = sum(products_counts) / len(products_counts)
    min_products = min(products_counts)
    max_products = max(products_counts)
    
    print(f"Products per query statistics:")
    print(f"Average products per query: {avg_products:.2f}")
    print(f"Minimum products per query: {min_products}")
    print(f"Maximum products per query: {max_products}")
    
    # Distribution of products per query
    count_distribution = Counter(products_counts)
    print("\nDistribution of products per query:")
    for count, frequency in sorted(count_distribution.items())[:10]:  # Show top 10
        print(f"{count} products: {frequency} queries ({frequency/len(products_per_query)*100:.2f}%)")

def analyze_esci_labels(data):
    """Analyze ESCI labels"""
    print("\n=== ESCI Label Analysis ===")
    
    # Count label distribution
    label_counts = Counter()
    label_counts_by_locale = defaultdict(Counter)
    
    for row in data:
        label = row['esci_label']
        locale = row['query_locale']
        
        label_counts[label] += 1
        label_counts_by_locale[locale][label] += 1
    
    # Overall distribution
    total = sum(label_counts.values())
    print("Overall label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count} ({count/total*100:.2f}%)")
    
    # Distribution by locale
    print("\nLabel distribution by locale:")
    for locale, counts in label_counts_by_locale.items():
        locale_total = sum(counts.values())
        print(f"\n{locale}:")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count} ({count/locale_total*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Analyze the shopping queries dataset CSV')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('--max_rows', type=int, default=None, help='Maximum number of rows to analyze')
    args = parser.parse_args()
    
    # Load data
    data = load_csv_data(args.csv_path, args.max_rows)
    
    # Analyze data
    analyze_queries(data)
    analyze_products(data)
    analyze_query_product_relationships(data)
    analyze_esci_labels(data)
    
    print("\nAnalysis completed.")

if __name__ == "__main__":
    main()
