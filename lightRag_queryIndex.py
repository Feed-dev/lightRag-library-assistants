from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded working directory
WORKING_DIR = "./rag_indices/pdf_library_test1_index"


class RAGQueryChat:
    def __init__(self):
        """Initialize RAG query system with hardcoded index."""
        try:
            self.rag = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=gpt_4o_mini_complete
            )
            logger.info(f"Successfully loaded index from {WORKING_DIR}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise

    def query(self, question: str) -> str:
        """Execute query using hybrid mode."""
        try:
            response = self.rag.query(
                question,
                param=QueryParam(
                    mode="hybrid",
                    max_token_for_text_unit=4000
                )
            )
            return response
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Error processing query: {str(e)}"

    def chat_loop(self):
        """Main chat loop."""
        print(f"Welcome to RAG Query Chat!")
        print("Type 'quit' to exit")

        while True:
            try:
                user_input = input("\nQuestion: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break

                print("\nSearching...")
                response = self.query(user_input)
                print("\nResponse:", response)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"An error occurred: {str(e)}")


def main():
    """Main entry point."""
    try:
        chat = RAGQueryChat()
        chat.chat_loop()
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
