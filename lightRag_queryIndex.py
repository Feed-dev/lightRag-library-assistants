from lightrag import LightRAG, QueryParam
import logging
import os
import json

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGQuerier:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.rag = None
        self.initialize_rag()

    def initialize_rag(self):
        try:
            if not os.path.exists(self.index_path):
                logger.error(f"Index directory not found: {self.index_path}")
                return False

            self.rag = LightRAG(working_dir=self.index_path)
            logger.info(f"Successfully loaded index from: {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            return False

    def search(self, query: str, mode: str = "hybrid") -> str:
        try:
            if not self.rag:
                return "No index loaded"

            # Create query parameters
            param = QueryParam(
                mode=mode,
                top_k=60,
                max_token_for_text_unit=4000,
                max_token_for_global_context=4000,
                max_token_for_local_context=4000,
                response_type="Multiple Paragraphs"
            )

            # Execute search
            result = self.rag.query(query=query, param=param)

            if result is None:
                return "No results found for the query"

            return result

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search failed: {str(e)}"


def main():
    # Update this path to match your index location
    index_path = "./rag_indices/pdf_library_index"

    # Initialize querier
    querier = RAGQuerier(index_path)

    # Test different search modes
    modes = ["hybrid", "local", "global", "naive"]
    query = "What are the main topics covered in the documents?"

    for mode in modes:
        print(f"\nTrying search mode: {mode}")
        print("-" * 80)

        result = querier.search(query, mode=mode)
        if result and result != "No results found for the query":
            print("Result:", result)
            print("-" * 80)
            break
        else:
            print(f"No results in {mode} mode, trying next mode...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSearch interrupted by user")
    except Exception as e:
        logger.error(f"Error during search: {e}")
