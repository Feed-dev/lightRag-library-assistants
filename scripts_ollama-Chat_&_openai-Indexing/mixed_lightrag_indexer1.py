import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import spacy
import networkx as nx
from pyvis.network import Network
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, ollama_embedding
from lightrag.utils import compute_mdhash_id, EmbeddingFunc
import logging
from typing import List, Optional, Generator, Literal, Dict
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

# Constants
RPM_LIMIT = 4000  # 80% of max limit
TPM_LIMIT = 1600000  # 80% of max limit
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
BATCH_SIZE = 32
CHUNK_TOKEN_SIZE = 512  # Optimized for GPT-4o-mini
CHUNK_OVERLAP_SIZE = 50  # ~10% overlap
EMBEDDING_DIM = 768  # nomic-embed-text dimension
MAX_TOKEN_SIZE = 8192  # Context window size

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
            self.requests = [(req_time, count) for req_time, count in self.requests
                             if req_time > minute_ago]
            self.tokens = [(token_time, count) for token_time, count in self.tokens
                           if token_time > minute_ago]

            if len(self.requests) >= self.rpm_limit:
                sleep_time = self.requests[0][0] - minute_ago + 0.1
                time.sleep(sleep_time)

            if sum(count for _, count in self.tokens) + num_tokens >= self.tpm_limit:
                sleep_time = self.tokens[0][0] - minute_ago + 0.1
                time.sleep(sleep_time)

            self.requests.append((now, 1))
            if num_tokens > 0:
                self.tokens.append((now, num_tokens))

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def extract_text_from_page(page) -> str:
    """Extract text from PDF page with improved UTF-8 handling"""
    try:
        # Try direct text extraction first
        text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES |
                                           fitz.TEXT_PRESERVE_WHITESPACE)
        text = text.encode('utf-8', errors='ignore').decode('utf-8')

        if text.strip():
            return text

        # Fallback to OCR if needed
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)
        return text.encode('utf-8', errors='ignore').decode('utf-8')

    except Exception as e:
        logger.error(f"Error extracting text from page: {e}")
        return ""


class ProcessingState:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.processed_files = set()
        self.failed_files = {}
        self.last_successful_batch = 0
        self.load_state()

    def save_state(self):
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'processed': list(self.processed_files),
                'failed': self.failed_files,
                'last_batch': self.last_successful_batch
            }, f, ensure_ascii=False)

    def load_state(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.processed_files = set(data.get('processed', []))
                self.failed_files = data.get('failed', {})
                self.last_successful_batch = data.get('last_batch', 0)


class LightRAGManager:
    def __init__(self, base_dir: str = "./rag_indices"):
        self.base_dir = base_dir
        self.current_working_dir = None
        self.rag = None
        self.rate_limiter = RateLimiter()

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def chunk_text(self, text: str) -> Generator[Dict, None, None]:
        """Chunk text using optimized token-based chunking"""
        # Clean and normalize text
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')

        doc = nlp(text)
        current_chunk = []
        current_size = 0

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Get token count for the sentence
            sent_tokens = len(sent)

            if current_size + sent_tokens > self.rag.chunk_token_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    yield {
                        "content": chunk_text,
                        "metadata": {
                            "chunk_size": len(chunk_text),
                            "token_count": current_size
                        }
                    }
                current_chunk = [sent_text]
                current_size = sent_tokens
            else:
                current_chunk.append(sent_text)
                current_size += sent_tokens

        # Yield last chunk if it exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            yield {
                "content": chunk_text,
                "metadata": {
                    "chunk_size": len(chunk_text),
                    "token_count": current_size
                }
            }

    def _process_pdf_with_chunks(self, file_path: str) -> List[Dict]:
        chunks = []
        try:
            doc = fitz.open(file_path)
            doc_metadata = doc.metadata

            for page_num in tqdm(range(len(doc)), desc=f"Processing {os.path.basename(file_path)}"):
                page = doc.load_page(page_num)
                page_text = extract_text_from_page(page)

                if page_text:
                    page_chunks = list(self.chunk_text(page_text))

                    for chunk in page_chunks:
                        if chunk["content"].strip():  # Verify chunk has content
                            chunk_metadata = {
                                "source": file_path,
                                "page": page_num + 1,
                                "title": doc_metadata.get("title", ""),
                                "author": doc_metadata.get("author", ""),
                                **chunk["metadata"]
                            }
                            chunks.append({
                                "content": chunk["content"],
                                "metadata": chunk_metadata
                            })

                page = None
                gc.collect()

            doc.close()
            return chunks

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []

    def _safe_batch_insert(self, chunks: List[Dict]) -> None:
        try:
            if not chunks:
                logger.warning("Empty chunk list received")
                return

            # Process in optimal batch sizes
            batch_size = min(len(chunks), self.rag.embedding_batch_num)

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                # Prepare batch for insertion
                batch_docs = {}
                for chunk in batch:
                    chunk_id = compute_mdhash_id(chunk["content"], prefix="chunk-")
                    batch_docs[chunk_id] = {
                        "content": chunk["content"],
                        "metadata": chunk["metadata"]
                    }

                self.rate_limiter.wait_if_needed()
                self.rag.insert(batch_docs)

        except Exception as e:
            logger.error(f"Batch insertion failed: {e}")
            raise

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

    def _initialize_lightrag(self, working_dir: str) -> LightRAG:
        """Common initialization for both create_index and switch_index"""
        return LightRAG(
            working_dir=working_dir,
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=MAX_TOKEN_SIZE,
                func=lambda texts: ollama_embedding(
                    texts,
                    embed_model="nomic-embed-text"
                )
            ),
            chunk_token_size=CHUNK_TOKEN_SIZE,
            chunk_overlap_token_size=CHUNK_OVERLAP_SIZE,
            embedding_batch_num=BATCH_SIZE,
            tiktoken_model_name="gpt-4o-mini"
        )

    def create_index(self, index_name: str) -> bool:
        working_dir = os.path.join(self.base_dir, index_name)
        if os.path.exists(working_dir):
            logger.warning(f"Index {index_name} already exists")
            return False
        os.makedirs(working_dir)
        self.current_working_dir = working_dir
        self.rag = self._initialize_lightrag(working_dir)
        logger.info(f"Created new index: {index_name}")
        return True

    def switch_index(self, index_name: str) -> bool:
        working_dir = os.path.join(self.base_dir, index_name)
        if not os.path.exists(working_dir):
            logger.error(f"Index {index_name} does not exist")
            return False
        self.current_working_dir = working_dir
        self.rag = self._initialize_lightrag(working_dir)
        logger.info(f"Switched to index: {index_name}")
        return True

    def list_indices(self) -> List[str]:
        """List all available indices."""
        return [d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))]

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
        if not self.rag:
            logger.error("No index selected")
            return

        state = ProcessingState(os.path.join(self.current_working_dir, 'processing_state.json'))

        pdf_files = [f for f in os.listdir(directory_path)
                     if f.lower().endswith('.pdf') and
                     f not in state.processed_files]

        with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
            for file in pdf_files:
                try:
                    file_path = os.path.join(directory_path, file)
                    if not self._validate_pdf(file_path):
                        continue

                    file_chunks = self._process_pdf_with_chunks(file_path)
                    if file_chunks:
                        for i in range(0, len(file_chunks), BATCH_SIZE):
                            batch = file_chunks[i:i + BATCH_SIZE]
                            self._safe_batch_insert(batch)
                            gc.collect()

                    state.processed_files.add(file)
                    state.save_state()
                    pbar.update(1)

                except Exception as e:
                    state.failed_files[file] = str(e)
                    logger.error(f"Error processing {file}: {e}")
                    continue

    def incremental_insert_pdf(self, file_path: str) -> bool:
        if not self.rag:
            logger.error("No index selected")
            return False

        try:
            if not self._validate_pdf(file_path):
                return False

            # Process PDF into chunks with metadata
            chunks = self._process_pdf_with_chunks(file_path)
            if not chunks:
                logger.warning(f"No valid text extracted from {file_path}")
                return False

            # Insert chunks
            self._safe_batch_insert(chunks)

            logger.info(f"Successfully processed {len(chunks)} chunks from: {file_path}")
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
    """Main function focused on PDF processing and index management."""
    try:
        # Initialize the manager
        logger.info("Initializing LightRAG Manager...")
        manager = LightRAGManager()

        # Demonstrate index management
        index_operations(manager)

        # Process PDFs if index is ready
        if manager.rag:
            process_pdf_library(manager)

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


def index_operations(manager: LightRAGManager) -> None:
    """Handle index creation, listing, and switching."""
    try:
        # List existing indices
        existing_indices = manager.list_indices()
        logger.info(f"Existing indices: {existing_indices}")

        # Create or switch to index
        index_name = "add your index name"  # add your index name
        if index_name in existing_indices:
            logger.info(f"Switching to existing index: {index_name}")
            success = manager.switch_index(index_name)
        else:
            logger.info(f"Creating new index: {index_name}")
            success = manager.create_index(index_name)

        if not success:
            logger.error("Failed to initialize index")
            raise RuntimeError("Index initialization failed")

    except Exception as e:
        logger.error(f"Index operations failed: {e}")
        raise


def process_pdf_library(manager: LightRAGManager) -> None:
    """Process PDF library with proper error handling and logging."""
    try:
        # Configure your PDF directory
        pdf_directory = r"path/to/your/pdf/library"  # path/to/your/pdf/library
        if not os.path.exists(pdf_directory):
            raise FileNotFoundError(f"Directory not found: {pdf_directory}")

        # Process directory of PDFs
        logger.info(f"Starting batch processing of directory: {pdf_directory}")
        manager.batch_insert_pdfs(pdf_directory)

        # Verify processing results
        state = ProcessingState(os.path.join(manager.current_working_dir, 'processing_state.json'))
        logger.info(f"Successfully processed {len(state.processed_files)} files")

        if state.failed_files:
            logger.warning(f"Failed to process {len(state.failed_files)} files")
            for file, error in state.failed_files.items():
                logger.warning(f"Failed file: {file}, Error: {error}")

    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
        logger.info("Indexing process completed successfully")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application terminated")
