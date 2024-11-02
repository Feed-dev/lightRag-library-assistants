Here's the formatted instructions in markdown:

# Vector Embedding Extraction Guide

## Setup and Installation

```python
# Required imports
import json
import numpy as np
import matplotlib.pyplot as plt
```

## Core Functions

### Matrix Decoder
```python
def decode_matrix_string(matrix_string):
    """Convert the matrix string into a numpy array of embeddings"""
    try:
        vector_data = np.fromstring(matrix_string, sep=' ')
        num_vectors = len(vector_data) // 1536
        embeddings = vector_data.reshape(num_vectors, 1536)
        return embeddings
    except Exception as e:
        print(f"Error decoding matrix: {e}")
        return None
```

### Embedding Extractor
```python
def extract_embeddings(file_path):
    """Extract embeddings from a vector database file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        matrix = data.get('matrix', '')
        documents = data.get('data', [])
        embedding_dim = data.get('embedding_dim')
        
        embeddings = decode_matrix_string(matrix)
        
        if embeddings is not None:
            print(f"\nFile: {file_path}")
            print(f"Number of embeddings: {len(embeddings)}")
            print(f"Embedding dimension: {embedding_dim}")
            print(f"Sample embedding shape: {embeddings[0].shape}")
            print("\nFirst embedding vector (first 10 values):")
            print(embeddings[0][:10])
            
            embedding_map = {
                doc['__id__']: embeddings[i] 
                for i, doc in enumerate(documents)
            }
            
            return embedding_map
            
    except Exception as e:
        print(f"Error processing file: {e}")
        return None
```

## Usage Examples

### Process Vector Database Files
```python
files = [
    'vdb_relationships.json',
    'vdb_chunks.json', 
    'vdb_entities.json'
]

for file in files:
    embedding_map = extract_embeddings(file)
```

### Access Individual Embeddings
```python
def access_embedding(embedding_map, doc_id):
    if doc_id in embedding_map:
        embedding = embedding_map[doc_id]
        print(f"Embedding for {doc_id}:")
        print(embedding[:10])  # Show first 10 values
```

### Statistical Analysis
```python
def analyze_embedding(embedding):
    print(f"Mean: {np.mean(embedding)}")
    print(f"Std: {np.std(embedding)}")
    print(f"Min: {np.min(embedding)}")
    print(f"Max: {np.max(embedding)}")
```

### Visualization
```python
def plot_embedding_distribution(embedding):
    plt.figure(figsize=(10, 5))
    plt.hist(embedding, bins=50)
    plt.title("Embedding Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
```

## File Structure
The embeddings are stored in JSON files with:
- `embedding_dim`: 1536 (OpenAI embedding dimension)
- `data`: Array of documents with IDs
- `matrix`: Space-separated string of floating-point numbers

## Processing Steps
1. Load JSON file
2. Extract matrix string
3. Convert to numpy array
4. Reshape to [num_vectors, 1536]
5. Map document IDs to embeddings
6. Analyze or visualize as needed

## Notes
- Designed for OpenAI embeddings (1536 dimensions)
- Handles multiple vector database files
- Provides analysis and visualization tools
- Maps embeddings back to original document IDs
