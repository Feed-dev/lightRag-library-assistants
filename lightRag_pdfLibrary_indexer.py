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
from typing import List, Optional, Generator, Literal
import shutil
import json
import gc
import time
import threading
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Constants for Tier 1 Rate Limits
RPM_LIMIT = 500  # Requests per minute
TPM_LIMIT = 200000  # Tokens per minute
BATCH_QUEUE_LIMIT = 2000000  # Batch queue limit
MAX_CHUNK_SIZE = 500000
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
BATCH_SIZE = 50

# Type definitions
SearchMode = Literal["local", "global", "hybrid", "naive"]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load spaCy
nlp = spacy.load("en_core_web_sm")


class RateLimiter:
    def __init__(self, rpm_limit: int = RPM_LIMIT, tpm_limit: int = TPM_LIMIT):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.requests = []
        self.tokens = []
        self.lock = threading.Lock()

    def wait_if_needed(self, num_tokens: int = 0) -> None:
        with self.lock:
            now = time.time()
            minute_ago = now - 60

            # Clean old requests and tokens
            self.requests = [(req_time, count) for req_time, count in self.requests
                             if req_time > minute_ago]
            self.tokens = [(token_time, count) for token_time, count in self.tokens
                           if token_time > minute_ago]

            # Check if we're over the limits
            if len(self.requests) >= self.rpm_limit:
                sleep_time = self.requests[0][0] - minute_ago + 0.1
                time.sleep(sleep_time)

            if sum(count for _, count in self.tokens) + num_tokens >= self.tpm_limit:
                sleep_time = self.tokens[0][0] - minute_ago + 0.1
                time.sleep(sleep_time)

            # Add current request
            self.requests.append((now, 1))
            if num_tokens > 0:
                self.tokens.append((now, num_tokens))


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def process_with_backoff(func, *args, **kwargs):
    return func(*args, **kwargs)


def chunk_text(text: str, chunk_size: int = MAX_CHUNK_SIZE) -> Generator[str, None, None]:
    doc = nlp(text)
    current_chunk = []
    current_size = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_size = len(sent_text)

        if current_size + sent_size > chunk_size:
            yield ' '.join(current_chunk)
            current_chunk = [sent_text]
            current_size = sent_size
        else:
            current_chunk.append(sent_text)
            current_size += sent_size

    if current_chunk:
        yield ' '.join(current_chunk)


class ProcessingState:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.processed_files = set()
        self.failed_files = {}
        self.last_successful_batch = 0
        self.load_state()

    def save_state(self):
        with open(self.save_path, 'w') as f:
            json.dump({
                'processed': list(self.processed_files),
                'failed': self.failed_files,
                'last_batch': self.last_successful_batch
            }, f)

    def load_state(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                data = json.load(f)
                self.processed_files = set(data.get('processed', []))
                self.failed_files = data.get('failed', {})
                self.last_successful_batch = data.get('last_batch', 0)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def extract_text_from_page(page):
    try:
        text = page.get_text()
        if text.strip():
            return text
        else:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            return text
    except Exception as e:
        logger.error(f"Error extracting text from page: {e}")
        return ""


def preprocess_text(text: str) -> str:
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    doc = nlp(text)
    cleaned_sentences = []

    for sent in doc.sents:
        cleaned_words = [token.text for token in sent if not token.is_space]
        cleaned_sentence = ' '.join(cleaned_words)
        if cleaned_sentence:
            cleaned_sentences.append(cleaned_sentence)

    gc.collect()  # Force garbage collection
    return ' '.join(cleaned_sentences)


class LightRAGManager:
    def __init__(self, base_dir: str = "./rag_indices"):
        self.base_dir = base_dir
        self.current_working_dir = None
        self.rag = None
        self.rate_limiter = RateLimiter()

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def _validate_pdf(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False

        if os.path.getsize(file_path) > MAX_FILE_SIZE:
            logger.error(f"File too large: {file_path}")
            return False

        try:
            doc = fitz.open(file_path)
            doc.close()
            return True
        except Exception as e:
            logger.error(f"Invalid PDF file {file_path}: {e}")
            return False

    def _process_pdf_with_chunks(self, file_path: str) -> List[str]:
        chunks = []
        try:
            doc = fitz.open(file_path)

            for page_num in tqdm(range(len(doc)), desc=f"Processing {os.path.basename(file_path)}"):
                page = doc.load_page(page_num)
                page_text = extract_text_from_page(page)

                if page_text:
                    for chunk in chunk_text(page_text):
                        chunks.append(chunk)

                page = None
                gc.collect()

            doc.close()
            return chunks

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []

    def _safe_batch_insert(self, chunks: List[str]) -> None:
        """Safely insert chunks with rate limiting and retry logic."""
        try:
            estimated_tokens = sum(len(chunk.split()) * 1.3 for chunk in chunks)
            self.rate_limiter.wait_if_needed(int(estimated_tokens))

            process_with_backoff(self.rag.insert, chunks)
        except Exception as e:
            logger.error(f"Batch insertion failed: {e}")
            raise

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

    def batch_insert_pdfs(self, directory_path: str) -> None:
        """Process and batch insert PDFs with rate limiting."""
        if not self.rag:
            logger.error("No index selected")
            return

        state = ProcessingState(os.path.join(self.current_working_dir, 'processing_state.json'))

        pdf_files = [f for f in os.listdir(directory_path)
                     if f.lower().endswith('.pdf') and
                     f not in state.processed_files]

        chunks = []
        with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
            for file in pdf_files:
                try:
                    if len(chunks) >= BATCH_SIZE:
                        self._safe_batch_insert(chunks)
                        chunks = []
                        gc.collect()

                    file_path = os.path.join(directory_path, file)
                    if not self._validate_pdf(file_path):
                        continue

                    file_chunks = self._process_pdf_with_chunks(file_path)
                    chunks.extend(file_chunks)

                    state.processed_files.add(file)
                    state.save_state()
                    pbar.update(1)

                except Exception as e:
                    state.failed_files[file] = str(e)
                    logger.error(f"Error processing {file}: {e}")
                    continue

            # Process remaining chunks
            if chunks:
                self._safe_batch_insert(chunks)

    def incremental_insert_pdf(self, file_path: str) -> bool:
        """Process and insert a single PDF with validation and chunking."""
        if not self.rag:
            logger.error("No index selected")
            return False

        try:
            if not self._validate_pdf(file_path):
                logger.error(f"PDF validation failed for {file_path}")
                return False

            with tqdm(total=1, desc=f"Processing {os.path.basename(file_path)}") as pbar:
                chunks = self._process_pdf_with_chunks(file_path)

                if not chunks:
                    logger.warning(f"No valid text extracted from {file_path}")
                    return False

                for i in range(0, len(chunks), BATCH_SIZE):
                    batch = chunks[i:i + BATCH_SIZE]
                    self._safe_batch_insert(batch)

                pbar.update(1)

            logger.info(f"Successfully inserted {len(chunks)} chunks from: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False

    def search(self, query: str, mode: SearchMode = "hybrid") -> str:
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

        try:
            G = nx.read_graphml(graph_path)
            net = Network(notebook=True)
            net.from_nx(G)

            if output_path is None:
                output_path = f"graph_{os.path.basename(self.current_working_dir)}.html"

            net.show(output_path)
            return f"Graph visualization saved to {output_path}"
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return f"Failed to generate visualization: {str(e)}"


def main():
    """Main function demonstrating LightRAG functionality with enhanced error handling and monitoring."""
    try:
        # Initialize the manager with progress tracking
        logger.info("Initializing LightRAG Manager...")
        manager = LightRAGManager()

        # Create or switch to index with validation
        index_name = "my_test_library_index"
        if index_name in manager.list_indices():
            logger.info(f"Switching to existing index: {index_name}")
            success = manager.switch_index(index_name)
        else:
            logger.info(f"Creating new index: {index_name}")
            success = manager.create_index(index_name)

        if not success:
            logger.error("Failed to initialize index")
            return

        # Process directory of PDFs with progress tracking
        pdf_directory = r"your library directory"  # fill in the path to your library directory
        logger.info(f"Starting batch processing of directory: {pdf_directory}")

        if not os.path.exists(pdf_directory):
            logger.error(f"Directory not found: {pdf_directory}")
            return

        try:
            manager.batch_insert_pdfs(pdf_directory)
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Continue execution even if batch processing fails

        # Demonstrate incremental PDF insertion with enhanced validation
        single_pdf = r"your pdf"  # fill in the path to your pdf

        if os.path.exists(single_pdf):
            logger.info("Processing single PDF...")
            try:
                success = manager.incremental_insert_pdf(single_pdf)
                if not success:
                    logger.warning("Failed to process single PDF")
            except Exception as e:
                logger.error(f"Single PDF processing failed: {e}")
        else:
            logger.error(f"File not found: {single_pdf}")

        # Demonstrate search capabilities with error handling
        logger.info("Performing search tests...")
        query = "What is astral projection?"
        search_modes = ["naive", "local", "global", "hybrid"]  # Type hint moved to SearchMode definition

        search_results = {}
        for mode in search_modes:
            try:
                result = manager.search(query, mode=mode)
                search_results[mode] = result
                print(f"\n{mode.capitalize()} search results:")
                print("-" * 50)
                print(result)
                print("-" * 50)
            except Exception as e:
                logger.error(f"Search failed for mode {mode}: {e}")
                search_results[mode] = f"Search failed: {str(e)}"

        # Generate and save graph visualization with error handling
        logger.info("Generating graph visualization...")
        try:
            output_path = os.path.join(manager.current_working_dir, "rag_visualization.html")
            result = manager.visualize_graph(output_path)
            print(f"\nVisualization: {result}")
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")

        return search_results  # Return search results for potential further analysis

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    try:
        results = main()
        if results:
            logger.info("Search results saved successfully")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application terminated")
