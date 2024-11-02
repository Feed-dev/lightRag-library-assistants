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



Here's a complete script for Vector Embedding Extraction that you can save and run:

# vector_extractor.md

```markdown
# Vector Embedding Extraction Script

## Setup

```python
# vector_extractor.py
import json
import numpy as np
import os

class VectorExtractor:
    def __init__(self, working_dir="."):
        self.working_dir = working_dir
        
    def decode_matrix_string(self, matrix_string):
        """Convert the matrix string into a numpy array of embeddings"""
        try:
            if not matrix_string:
                return np.empty((0, 1536))
            
            # Convert string of numbers into numpy array
            vector_data = np.fromstring(matrix_string, sep=' ')
            
            # Reshape based on embedding dimension (1536 for OpenAI embeddings)
            num_vectors = len(vector_data) // 1536
            embeddings = vector_data.reshape(num_vectors, 1536)
            
            return embeddings
        except Exception as e:
            print(f"Error decoding matrix: {e}")
            return None

    def extract_embeddings(self, file_path):
        """Extract embeddings from a vector database file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract matrix string and document IDs
            matrix = data.get('matrix', '')
            documents = data.get('data', [])
            embedding_dim = data.get('embedding_dim')
            
            # Decode matrix into numpy arrays
            embeddings = self.decode_matrix_string(matrix)
            
            if embeddings is not None:
                print(f"\nFile: {file_path}")
                print(f"Number of embeddings: {len(embeddings)}")
                print(f"Embedding dimension: {embedding_dim}")
                if len(embeddings) > 0:
                    print(f"Sample embedding shape: {embeddings.shape}")
                    print("\nFirst embedding vector (first 10 values):")
                    print(embeddings[:10])
                
                # Create mapping of document IDs to embeddings
                embedding_map = {
                    doc['__id__']: embeddings[i] 
                    for i, doc in enumerate(documents)
                }
                
                return embedding_map
                
        except Exception as e:
            print(f"Error processing file: {e}")
            return None

    def process_files(self):
        """Process all vector database files"""
        files = [
            'vdb_relationships.json',
            'vdb_chunks.json',
            'vdb_entities.json'
        ]
        
        results = {}
        for file in files:
            file_path = os.path.join(self.working_dir, file)
            if os.path.exists(file_path):
                results[file] = self.extract_embeddings(file_path)
            else:
                print(f"File not found: {file_path}")
                
        return results

def main():
    # Initialize extractor
    extractor = VectorExtractor()
    
    # Process files and get embeddings
    embeddings = extractor.process_files()
    
    # Example: Access embeddings for a specific document
    if embeddings.get('vdb_relationships.json'):
        rel_embeddings = embeddings['vdb_relationships.json']
        # Get first document ID
        first_doc_id = list(rel_embeddings.keys())
        print(f"\nEmbedding for document {first_doc_id}:")
        print(rel_embeddings[first_doc_id][:10])  # Show first 10 values

if __name__ == "__main__":
    main()
```

## Usage

1. Save the script as `vector_extractor.py`

2. Place it in the same directory as your vector database files:
```
working_dir/
    ├── vector_extractor.py
    ├── vdb_relationships.json
    ├── vdb_chunks.json
    └── vdb_entities.json
```

3. Run the script:
```bash
python vector_extractor.py
```

## Additional Features

You can extend the VectorExtractor class with additional methods:

```python
def analyze_embeddings(self, embeddings):
    """Analyze statistical properties of embeddings"""
    if embeddings is None:
        return
        
    for doc_id, embedding in embeddings.items():
        print(f"\nDocument: {doc_id}")
        print(f"Mean: {np.mean(embedding)}")
        print(f"Std: {np.std(embedding)}")
        print(f"Min: {np.min(embedding)}")
        print(f"Max: {np.max(embedding)}")

def save_embeddings(self, embeddings, output_file):
    """Save embeddings to a numpy file"""
    if embeddings is None:
        return
        
    np.save(output_file, embeddings)
```

## Error Handling

The script includes error handling for:
- Missing files
- Invalid JSON format
- Matrix decoding errors
- Empty or malformed embeddings

## Output Format

The script returns a dictionary with:
- File names as keys
- Embedding maps as values, where each map contains:
  - Document IDs as keys
  - Numpy arrays (1536-dimensional vectors) as values

This format makes it easy to:
1. Access specific document embeddings
2. Perform vector operations
3. Export to other formats
4. Use with machine learning models
