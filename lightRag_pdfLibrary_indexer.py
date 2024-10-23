import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import spacy
import networkx as nx
from pyvis.network import Network
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
import logging
from typing import List, Optional
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure pytesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def extract_text_from_page(page):
    try:
        text = page.get_text()
        if text.strip():  # If there's text, it's not a scan
            return text
        else:  # If no text, it's likely a scan
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            return text
    except Exception as e:
        logger.error(f"Error extracting text from page: {e}")
        return ""


def preprocess_text(text):
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    doc = nlp(text)
    cleaned_sentences = []
    for sent in doc.sents:
        cleaned_words = [token.text for token in sent if not token.is_space]
        cleaned_sentence = ' '.join(cleaned_words)
        if cleaned_sentence:
            cleaned_sentences.append(cleaned_sentence)
    return ' '.join(cleaned_sentences)


def process_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = extract_text_from_page(page)
            if not page_text.strip():
                logger.warning(f"No text extracted from page {page_num} of {file_path}")
                continue
            full_text += page_text + "\n\n"
        doc.close()
        return preprocess_text(full_text)
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return ""


class LightRAGManager:
    def __init__(self, base_dir: str = "./rag_indices"):
        self.base_dir = base_dir
        self.current_working_dir = None
        self.rag = None
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def create_index(self, index_name: str) -> bool:
        """Create a new index with specified name."""
        working_dir = os.path.join(self.base_dir, index_name)
        if os.path.exists(working_dir):
            logger.warning(f"Index {index_name} already exists")
            return False

        os.makedirs(working_dir)
        self.current_working_dir = working_dir
        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=gpt_4o_mini_complete,
        )
        logger.info(f"Created new index: {index_name}")
        return True

    def list_indices(self) -> List[str]:
        """List all available indices."""
        return [d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))]

    def switch_index(self, index_name: str) -> bool:
        """Switch to an existing index."""
        working_dir = os.path.join(self.base_dir, index_name)
        if not os.path.exists(working_dir):
            logger.error(f"Index {index_name} does not exist")
            return False

        self.current_working_dir = working_dir
        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=gpt_4o_mini_complete,
        )
        logger.info(f"Switched to index: {index_name}")
        return True

    def delete_index(self, index_name: str) -> bool:
        """Delete an existing index."""
        working_dir = os.path.join(self.base_dir, index_name)
        if not os.path.exists(working_dir):
            logger.error(f"Index {index_name} does not exist")
            return False

        shutil.rmtree(working_dir)
        if self.current_working_dir == working_dir:
            self.current_working_dir = None
            self.rag = None
        logger.info(f"Deleted index: {index_name}")
        return True

    def batch_insert_pdfs(self, directory_path: str):
        """Process and batch insert PDFs from directory."""
        if not self.rag:
            logger.error("No index selected")
            return

        texts = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    logger.info(f"Processing: {file_path}")
                    text = process_pdf(file_path)
                    if text:
                        texts.append(text)

        if texts:
            self.rag.insert(texts)
            logger.info(f"Batch inserted {len(texts)} documents")

    def incremental_insert_pdf(self, file_path: str):
        """Process and insert a single PDF."""
        if not self.rag:
            logger.error("No index selected")
            return

        logger.info(f"Processing: {file_path}")
        text = process_pdf(file_path)
        if text:
            self.rag.insert(text)
            logger.info(f"Inserted: {file_path}")

    def search(self, query: str, mode: str = "hybrid") -> str:
        """Perform a search using specified mode."""
        if not self.rag:
            return "No index selected"

        return self.rag.query(query, param=QueryParam(mode=mode))

    def visualize_graph(self, output_path: Optional[str] = None) -> str:
        """Generate and save graph visualization."""
        if not self.rag:
            return "No index selected"

        graph_path = os.path.join(self.current_working_dir, 'graph_chunk_entity_relation.graphml')
        if not os.path.exists(graph_path):
            return "Graph file not found"

        G = nx.read_graphml(graph_path)
        net = Network(notebook=True)
        net.from_nx(G)

        if output_path is None:
            output_path = f"graph_{os.path.basename(self.current_working_dir)}.html"

        net.show(output_path)
        return f"Graph visualization saved to {output_path}"


def main():
    # Initialize the manager
    manager = LightRAGManager()

    # Create a new index
    manager.create_index("my_library")

    # Process directory of PDFs
    manager.batch_insert_pdfs("./pdfs")

    # Add a single new PDF
    manager.incremental_insert_pdf("./new_document.pdf")

    # Perform searches
    query = "What are the main themes?"
    print("Naive search:", manager.search(query, mode="naive"))
    print("Local search:", manager.search(query, mode="local"))
    print("Global search:", manager.search(query, mode="global"))
    print("Hybrid search:", manager.search(query, mode="hybrid"))

    # Generate graph visualization
    manager.visualize_graph()


if __name__ == "__main__":
    main()
