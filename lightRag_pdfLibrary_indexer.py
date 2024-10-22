import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import spacy
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure pytesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

WORKING_DIR = "./my_library_index"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    # llm_model_func=gpt_4o_complete  # Uncomment this line if you want to use gpt-4
)


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


def index_pdf(file_path):
    logger.info(f"Processing: {file_path}")
    text = process_pdf(file_path)
    if text:
        rag.insert(text)
        logger.info(f"Indexed: {file_path}")
    else:
        logger.warning(f"No text extracted from {file_path}")


def index_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                index_pdf(file_path)


def main():
    library_path = "./my_library"

    # Index the entire library
    index_directory(library_path)

    # Perform queries using different search modes
    query = "What are the main themes in Charles Dickens' works?"

    print("Naive search:")
    print(rag.query(query, param=QueryParam(mode="naive")))

    print("\nLocal search:")
    print(rag.query(query, param=QueryParam(mode="local")))

    print("\nGlobal search:")
    print(rag.query(query, param=QueryParam(mode="global")))

    print("\nHybrid search:")
    print(rag.query(query, param=QueryParam(mode="hybrid")))


if __name__ == "__main__":
    main()