# LightRAG PDF Library Indexer

A Python tool for managing and querying PDF document libraries using LightRAG (Retrieval-Augmented Generation) with OpenAI's GPT models. This tool provides comprehensive PDF processing capabilities, including OCR for scanned documents, and implements multiple search modes through a managed index system.

## Features

### PDF Processing
- Text extraction from both digital and scanned PDFs
- OCR support for scanned documents using Tesseract
- Text preprocessing and cleaning using spaCy
- Support for nested directory structures
- Batch and incremental document processing

### Index Management
- Create and manage multiple document indices
- Switch between existing indices
- List available indices
- Delete indices
- Graph visualization of knowledge structures

### Search Capabilities
- Naive search mode
- Local search mode
- Global search mode
- Hybrid search mode

## Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Python Packages
- PyMuPDF (fitz)
- pytesseract
- Pillow
- spacy
- lightrag
- networkx
- pyvis

### Additional Requirements
1. Tesseract OCR:
   - Windows: Download and install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`

2. spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

3. OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Configuration

1. Set Tesseract path (Windows):
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

2. Configure logging level:
```python
logging.basicConfig(level=logging.INFO)
```

## Usage

### Basic Usage

```python
from lightrag_pdflibrary_indexer import LightRAGManager

# Initialize manager
manager = LightRAGManager()

# Create new index
manager.create_index("my_library")

# Process PDF directory
manager.batch_insert_pdfs("./pdfs")

# Add single PDF
manager.incremental_insert_pdf("./new_document.pdf")

# Search
result = manager.search("What are the main themes?", mode="hybrid")
```

### Index Management

```python
# List indices
indices = manager.list_indices()

# Switch index
manager.switch_index("another_library")

# Delete index
manager.delete_index("old_library")

# Visualize graph
manager.visualize_graph()
```

### Search Modes

```python
# Available search modes
manager.search(query, mode="naive")   # Basic text matching
manager.search(query, mode="local")   # Context-aware local search
manager.search(query, mode="global")  # Global knowledge search
manager.search(query, mode="hybrid")  # Combined search strategies
```

## Implementation Details

### PDF Processing Pipeline
1. Text extraction (digital/OCR)
2. Text preprocessing
3. Sentence cleaning
4. Batch/incremental indexing

### Index Management
- Indices stored in separate directories
- Graph-based knowledge representation
- Support for multiple concurrent indices

### Search Implementation
- Multiple search modes for different use cases
- Query parameter customization
- Hybrid search combining multiple strategies

## Notes and Limitations

- Large PDF collections may require significant processing time
- OCR processing is slower than digital text extraction
- Memory usage increases with graph visualization of large indices
- OpenAI API costs should be monitored

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License

## Acknowledgments

- Built using [LightRAG](https://github.com/HKUDS/LightRAG)
- Uses OpenAI's GPT models
- Incorporates various open-source libraries for PDF and text processing
