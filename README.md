# LightRAG PDF Library Assistant

A Python-based conversational AI system that uses LightRAG to process, index, and query PDF document libraries.

## Components

### 1. PDF Indexer (`lightRag_pdfLibrary_indexer.py`)
- Processes and indexes PDF documents
- Handles both digital and scanned PDFs (with OCR)
- Creates and manages document indices

### 2. Query Index (`lightRag_queryIndex.py`)
- Provides multiple search modes (hybrid, local, global, naive)

### 3. Chat Interface (`rag_chatbot.py`)
- Interactive conversational interface
- Uses hybrid search for optimal results

## Installation

```bash
pip install -r requirements.txt
```

### Additional Requirements
1. Tesseract OCR:
   Windows: Download and install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   Set Tesseract path (Windows):
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

2. spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

3. OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```


## Usage
1. Index Your PDFs (`lightRag_pdfLibrary_indexer.py`)
```python
index_name = "pdf_library_index"  # add your index name
```
```python
pdf_directory = r"C:\Users\feder\e-books\ML_test_Library"  # path/to/your/pdf/library
```
   
2. Test Querying the Index (`lightRag_queryIndex.py`)
```python
index_path = "./rag_indices/pdf_library_index" # your index path
```
   
3. Start Chat Interface
```bash
python rag_chatbot.py
```
   
## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License

## Acknowledgments

- Built using [LightRAG](https://github.com/HKUDS/LightRAG)
- Uses OpenAI's GPT models
- Incorporates various open-source libraries for PDF and text processing
