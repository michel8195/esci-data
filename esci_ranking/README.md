# ESCI Ranking: Two-Tower Model for Query-Product Matching

This package provides a modular implementation of a two-tower model for query-product ranking using the ESCI (Exact, Substitute, Complement, Irrelevant) dataset.

## Features

- **Two-Tower Architecture**: Separate encoders for queries and products with projection layers
- **Rich Product Representations**: Combines product title, brand, color, and features
- **Efficient Inference**: Supports caching of embeddings for faster repeated evaluation
- **Modular Design**: Clean separation of concerns following best practices
- **Command-line Interface**: Easy-to-use scripts for training and inference

## Package Structure

```
esci_ranking/
├── data/            # Data loading and processing
├── models/          # Model architecture definitions
├── train/           # Training utilities
├── inference/       # Inference utilities
├── evaluation/      # Evaluation metrics
├── utils/           # Utility functions
└── README.md        # This file
```

## Installation

```bash
# Install in development mode
pip install -e .
```

## Usage

### Training a Model

```bash
# Using the command-line script
esci-train /path/to/train-v0.3.csv /path/to/product_catalogue-v0.3.csv /path/to/save/model --locale us --batch_size 4 --num_epochs 3

# Or using Python
from esci_ranking.train.train_model import main
main()
```

### Running Inference

```bash
# Using the command-line script
esci-infer /path/to/test_public-v0.3.csv /path/to/product_catalogue-v0.3.csv /path/to/model /path/to/output.csv --locale us --batch_size 32 --use_cached_embeddings

# Or using Python
from esci_ranking.inference.run_inference import main
main()
```

### Python API Example

```python
import torch
from transformers import AutoTokenizer
from esci_ranking.models.two_tower import create_model, load_inference_encoders
from esci_ranking.data.dataset import load_data, create_data_loaders
from esci_ranking.train.trainer import TwoTowerTrainer
from esci_ranking.inference.predictor import TwoTowerPredictor

# Load data
train_df, val_df, product_catalog_df, test_df = load_data(
    'path/to/train.csv', 
    'path/to/product_catalog.csv',
    locale='us',
    test_csv_path='path/to/test.csv'
)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# Create data loaders
train_dataloader, val_dataloader, test_dataloader = create_data_loaders(
    train_df, val_df, product_catalog_df, tokenizer, 
    batch_size=4, max_length=256, test_df=test_df
)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model().to(device)

# Train model
trainer = TwoTowerTrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device=device,
    model_save_path='path/to/save/model'
)
model, train_losses, val_metrics = trainer.train(num_epochs=3)

# Run inference
query_encoder, product_encoder = load_inference_encoders('path/to/model')
query_encoder = query_encoder.to(device)
product_encoder = product_encoder.to(device)

predictor = TwoTowerPredictor(
    query_encoder=query_encoder,
    product_encoder=product_encoder,
    tokenizer=tokenizer,
    device=device
)

rankings_df = predictor.predict(test_df, product_catalog_df)
```

## Model Architecture

The two-tower model consists of:

1. **Query Tower**: Encodes query text into embeddings
2. **Product Tower**: Encodes product text into embeddings
3. **Projection Layers**: Maps both embeddings to a common space
4. **Similarity Computation**: Uses dot product to calculate relevance

Each tower uses a pre-trained transformer model (default: bert-base-multilingual-cased) with projection layers to map the embeddings to a common space.

## Evaluation Metrics

The model is evaluated using standard ranking metrics:

- **NDCG@10**: Normalized Discounted Cumulative Gain at 10
- **Recall@k**: Proportion of relevant items in top-k results

## License

This package is released under the Apache License 2.0, the same as the original ESCI dataset.
